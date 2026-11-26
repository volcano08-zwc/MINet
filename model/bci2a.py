"""DualBranchMINet for BCI Competition IV 2a.

The implementation follows model_v1.md. The DCT-PSCA weight MLP is a
configurable provisional choice because the original PSCA figure and exact
weight-network dimensions were not supplied. Product-Sum is exactly U * A + A.
"""

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SincFilterBank(nn.Module):
    """Shared differentiable bandpass FIRs with ordered, shared boundaries.

    Softmax distributes the remaining frequency range among the lower margin,
    band excess widths, and upper margin. Adding a minimum width to every band
    guarantees ordered bands throughout optimization.
    """

    def __init__(
        self,
        sampling_rate: float = 250,
        initial_edges_hz: Sequence[float] = (4, 8, 13, 20, 30, 40),
        kernel_size: int = 251,
        min_freq_hz: float = 0.5,
        max_freq_hz: Optional[float] = None,
        min_bandwidth_hz: float = 1.0,
        learnable_cutoffs: bool = True,
        window_function: str = "hamming",
        padding_mode: str = "reflect",
        num_bands: Optional[int] = None,
        boundary_parameterization: str = "shared_softmax_gaps",
        share_across_electrodes: bool = True,
        context_samples: int = 0,
    ):
        super().__init__()
        if sampling_rate <= 0 or kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError("Sinc sampling_rate must be positive and kernel_size positive and odd.")
        if boundary_parameterization != "shared_softmax_gaps":
            raise ValueError("Supported Sinc boundary_parameterization is shared_softmax_gaps.")
        if not share_across_electrodes:
            raise ValueError("This architecture shares Sinc filters across all electrodes.")
        if padding_mode not in {"reflect", "replicate", "constant"}:
            raise ValueError("Sinc padding_mode must be reflect, replicate, or constant.")
        edges = torch.as_tensor(initial_edges_hz, dtype=torch.float64)
        if edges.ndim != 1 or len(edges) < 2 or not torch.isfinite(edges).all():
            raise ValueError("Sinc initial_edges_hz must be a finite one-dimensional boundary list.")
        self.num_bands = len(edges) - 1
        if num_bands is not None and num_bands != self.num_bands:
            raise ValueError("num_bands must equal len(initial_edges_hz) - 1.")
        self.sampling_rate = float(sampling_rate)
        self.min_freq_hz = float(min_freq_hz)
        self.max_freq_hz = float(max_freq_hz if max_freq_hz is not None else sampling_rate / 2 - 1)
        self.min_bandwidth_hz = float(min_bandwidth_hz)
        self.kernel_size = int(kernel_size)
        self.padding_mode = padding_mode
        self.context_samples = int(context_samples)
        if not 0 <= self.min_freq_hz < self.max_freq_hz < sampling_rate / 2:
            raise ValueError("Sinc frequency bounds must lie in [0, Nyquist).")
        if self.min_bandwidth_hz <= 0:
            raise ValueError("Sinc min_bandwidth_hz must be positive.")
        if self.context_samples < 0:
            raise ValueError("Sinc context_samples must be nonnegative.")
        gaps = torch.cat((
            edges[:1] - self.min_freq_hz,
            edges.diff() - self.min_bandwidth_hz,
            self.max_freq_hz - edges[-1:],
        ))
        if not bool((gaps > 0).all()):
            raise ValueError("Initial edges must be strictly inside bounds, with widths above the minimum.")
        self.free_range_hz = self.max_freq_hz - self.min_freq_hz - self.num_bands * self.min_bandwidth_hz
        raw_gap_logits = gaps.log().to(torch.float32)
        if learnable_cutoffs:
            self.raw_gap_logits = nn.Parameter(raw_gap_logits)
        else:
            self.register_buffer("raw_gap_logits", raw_gap_logits)
        self.register_buffer("sample_offsets", torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2)
        windows = {
            "hamming": torch.hamming_window,
            "hann": torch.hann_window,
            "blackman": torch.blackman_window,
        }
        if window_function not in windows:
            raise ValueError(f"Unsupported Sinc window_function {window_function!r}.")
        self.register_buffer("window", windows[window_function](kernel_size, periodic=False))

    def cutoff_edges(self) -> Tensor:
        gaps = self.raw_gap_logits.softmax(dim=0) * self.free_range_hz
        width_offsets = torch.arange(self.num_bands + 1, device=gaps.device, dtype=gaps.dtype)
        return self.min_freq_hz + gaps.cumsum(dim=0)[:-1] + width_offsets * self.min_bandwidth_hz

    def cutoff_frequencies(self) -> Tensor:
        """Return [bands, 2] lower/upper cutoffs, in Hz."""
        edges = self.cutoff_edges()
        return torch.stack((edges[:-1], edges[1:]), dim=-1)

    def filter_kernels(self) -> Tensor:
        edges = self.cutoff_edges()[:, None] / self.sampling_rate
        lowpass = 2 * edges * torch.sinc(2 * edges * self.sample_offsets[None, :])
        return (lowpass[1:] - lowpass[:-1]) * self.window[None, :]

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Sinc input must have shape [batch, electrodes, samples].")
        batch_size, electrodes, samples = x.shape
        padding = self.kernel_size // 2
        if self.padding_mode == "reflect" and samples <= padding:
            raise ValueError("Reflect Sinc padding requires input length > kernel_size // 2.")
        independent_signals = x.reshape(batch_size * electrodes, 1, samples)
        padded = F.pad(independent_signals, (padding, padding), mode=self.padding_mode)
        kernels = self.filter_kernels().to(dtype=x.dtype).unsqueeze(1)
        filtered = F.conv1d(padded, kernels)
        return filtered.reshape(batch_size, electrodes, self.num_bands, samples).permute(0, 2, 1, 3)


class SlidingWindowPower(nn.Module):
    def __init__(self, window_size: int = 250, stride: int = 50, eps: float = 1e-6):
        super().__init__()
        if window_size < 1 or stride < 1 or eps <= 0:
            raise ValueError("Power window_size, stride, and eps must be positive.")
        self.window_size = window_size
        self.stride = stride
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[-1] < self.window_size:
            raise ValueError("Power input must be [batch, bands, electrodes, samples] with a full window.")
        windows = x.unfold(-1, self.window_size, self.stride)
        return (windows.square().mean(dim=-1) + self.eps).log()


@dataclass
class PSCAConfig:
    """Configurable DCT descriptor and provisional channel-weight network."""

    dct_orders: Sequence[int] = (1, 2, 3)
    dct_group_mapping: str = "interleaved"
    custom_channel_orders: Optional[Sequence[int]] = None
    include_dc: bool = False
    hidden_dims: Sequence[int] = (27,)
    activation: str = "relu"
    output_activation: str = "sigmoid"
    bias: bool = True


def _activation(name: str) -> nn.Module:
    activations = {"relu": nn.ReLU, "gelu": nn.GELU, "sigmoid": nn.Sigmoid, "identity": nn.Identity}
    if name not in activations:
        raise ValueError(f"Unsupported activation {name!r}.")
    return activations[name]()


class DCTPSCA(nn.Module):
    """One DCT-II coefficient per channel, followed by U * A + A."""

    def __init__(self, channels: int = 110, sequence_length: int = 16, config=None):
        super().__init__()
        if config is None:
            config = PSCAConfig()
        if isinstance(config, Mapping):
            config = PSCAConfig(**config)
        if channels < 1 or sequence_length < 1:
            raise ValueError("DCT channels and sequence_length must be positive.")
        orders = list(config.dct_orders)
        if not orders:
            raise ValueError("DCT dct_orders cannot be empty.")
        if config.dct_group_mapping in {"interleaved", "channel_index % 3"}:
            channel_orders = [orders[index % len(orders)] for index in range(channels)]
        elif config.dct_group_mapping == "contiguous":
            sizes = [channels // len(orders) + int(index < channels % len(orders)) for index in range(len(orders))]
            channel_orders = [order for order, size in zip(orders, sizes) for _ in range(size)]
        elif config.dct_group_mapping == "custom":
            channel_orders = list(config.custom_channel_orders or [])
            if len(channel_orders) != channels:
                raise ValueError("custom_channel_orders must contain one DCT order per channel.")
        else:
            raise ValueError("DCT group mapping must be interleaved, contiguous, or custom.")
        if any(int(order) != order or not 0 <= order < sequence_length for order in channel_orders):
            raise ValueError("DCT orders must be integer indices smaller than sequence_length.")
        if not config.include_dc and 0 in channel_orders:
            raise ValueError("DCT order zero requires include_dc=true.")
        channel_k = torch.tensor(channel_orders, dtype=torch.long)
        positions = torch.arange(sequence_length, dtype=torch.float64) + 0.5
        basis = torch.cos(math.pi * channel_k[:, None] * positions / sequence_length)
        scaling = torch.full((channels, 1), math.sqrt(2 / sequence_length), dtype=torch.float64)
        scaling[channel_k == 0] = math.sqrt(1 / sequence_length)
        self.channels = channels
        self.sequence_length = sequence_length
        self.register_buffer("channel_dct_k", channel_k)
        self.register_buffer("dct_basis", (basis * scaling).to(torch.float32))
        dims = [channels, *config.hidden_dims, channels]
        if any(dim < 1 for dim in dims):
            raise ValueError("PSCA hidden dimensions must be positive.")
        layers = []
        for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim, bias=config.bias))
            if index < len(dims) - 2:
                layers.append(_activation(config.activation))
        layers.append(_activation(config.output_activation))
        self.channel_weight = nn.Sequential(*layers)

    def describe(self, x: Tensor) -> Tensor:
        """Return the selected DCT-II coefficient for each feature channel."""
        if x.ndim != 3 or x.shape[1:] != (self.channels, self.sequence_length):
            raise ValueError(f"DCT input must be [batch, {self.channels}, {self.sequence_length}].")
        return (x * self.dct_basis.to(dtype=x.dtype)).sum(dim=-1)

    def forward(self, x: Tensor, return_weights: bool = False):
        weights = self.channel_weight(self.describe(x))
        output = x * weights.unsqueeze(-1) + weights.unsqueeze(-1)
        if return_weights:
            return output, weights
        return output


def _transformer(d_model: int, config: Mapping, dropout: float) -> nn.TransformerEncoder:
    heads = config["heads"]
    if heads < 1 or d_model % heads:
        raise ValueError("Transformer d_model must be divisible by heads.")
    if config["layers"] < 1:
        raise ValueError("Transformer layers must be positive.")
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=config["dim_feedforward"],
        dropout=dropout,
        activation=config.get("activation", "gelu"),
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(layer, num_layers=config["layers"], enable_nested_tensor=False)
    # TransformerEncoder clones initial parameters. Initialize each layer independently.
    for block in encoder.layers:
        nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
        nn.init.xavier_uniform_(block.linear1.weight)
        nn.init.xavier_uniform_(block.linear2.weight)
    return encoder


class FrequencyBranch(nn.Module):
    def __init__(self, config: Mapping):
        super().__init__()
        data_config = config["data"]
        model_config = config["model"]
        self.target_samples = round((data_config["epoch_end_sec"] - data_config["epoch_start_sec"]) * data_config["sampling_rate"])
        self.context_samples = model_config["sinc"]["context_samples"]
        self.filterbank = SincFilterBank(sampling_rate=data_config["sampling_rate"], **model_config["sinc"])
        self.power = SlidingWindowPower(**model_config["power"])
        self.n_windows = (self.target_samples - self.power.window_size) // self.power.stride + 1
        channels = self.filterbank.num_bands * data_config["n_channels"]
        self.psca = DCTPSCA(channels, self.n_windows, model_config["psca"])
        self.projection = nn.Linear(channels, model_config["d_model"])
        max_len = model_config["temporal_transformer"]["max_len"]
        if not 1 <= self.n_windows <= max_len:
            raise ValueError("Power window count must lie in [1, temporal max_len].")
        self.position_embedding = nn.Parameter(torch.empty(1, max_len, model_config["d_model"]))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.transformer = _transformer(model_config["d_model"], model_config["temporal_transformer"], model_config["dropout"])

    def forward(self, x: Tensor, return_aux: bool = False):
        expected_length = self.target_samples + 2 * self.context_samples
        if x.shape[-1] != expected_length:
            raise ValueError(f"Frequency input requires {expected_length} samples, including per-side context.")
        filtered = self.filterbank(x)
        target_filtered = filtered[..., self.context_samples:self.context_samples + self.target_samples]
        power = self.power(target_filtered)
        band_electrode = power.flatten(1, 2)
        attended, weights = self.psca(band_electrode, return_weights=True)
        tokens = self.projection(attended.transpose(1, 2))
        tokens = self.transformer(tokens + self.position_embedding[:, :tokens.shape[1]])
        if return_aux:
            return tokens, {
                "sinc_cutoffs": self.filterbank.cutoff_frequencies(),
                "filtered_signal": target_filtered,
                "power_features": power,
                "dct_descriptors": self.psca.describe(band_electrode),
                "psca_weights": weights,
                "psca_features": attended,
                "time_tokens": tokens,
            }
        return tokens


class SharedElectrodeEncoder(nn.Module):
    """Identical temporal Conv1d parameters applied to every EEG electrode."""

    def __init__(self, d_model: int = 64, config: Optional[Mapping] = None):
        super().__init__()
        if config is None:
            config = {
                "encoder_channels": [32, d_model], "encoder_kernel_sizes": [25, 15],
                "encoder_strides": [5, 2], "encoder_paddings": [12, 7],
            }
        channels = list(config["encoder_channels"])
        kernels = config["encoder_kernel_sizes"]
        strides = config["encoder_strides"]
        paddings = config["encoder_paddings"]
        if not channels or not len(channels) == len(kernels) == len(strides) == len(paddings):
            raise ValueError("Electrode encoder layer lists must have matching nonzero lengths.")
        if channels[-1] != d_model:
            raise ValueError("Last electrode encoder channel count must equal d_model.")
        layers = []
        input_channels = 1
        for output_channels, kernel, stride, padding in zip(channels, kernels, strides, paddings):
            layers.extend((
                nn.Conv1d(input_channels, output_channels, kernel, stride=stride, padding=padding),
                nn.BatchNorm1d(output_channels), nn.GELU(),
            ))
            input_channels = output_channels
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.encoder = nn.Sequential(*layers)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Electrode encoder input must be [batch, electrodes, samples].")
        batch_size, electrodes, samples = x.shape
        return self.encoder(x.reshape(batch_size * electrodes, 1, samples)).reshape(batch_size, electrodes, self.d_model)


def load_electrode_coordinates(channel_names: Sequence[str], spatial_config: Mapping) -> Tensor:
    """Read MNE's packaged ASA standard_1005 montage without importing MNE.

    The installed montage is an offline data dependency. ASA file coordinates
    are in millimetres and are converted to metres. Pairwise distances are
    invariant to the rigid transform MNE normally applies to this montage.
    """
    explicit = spatial_config.get("coordinates_m")
    if explicit is not None:
        coordinates = torch.as_tensor(explicit, dtype=torch.float32)
    else:
        if spatial_config.get("coordinate_source", "standard_1005") != "standard_1005":
            raise ValueError("Use coordinate_source=standard_1005 or provide coordinates_m.")
        mne_spec = importlib.util.find_spec("mne")
        if mne_spec is None or not mne_spec.submodule_search_locations:
            raise ImportError("Install mne for packaged standard_1005 coordinates, or configure coordinates_m.")
        montage_file = Path(next(iter(mne_spec.submodule_search_locations))) / "channels" / "data" / "montages" / "standard_1005.elc"
        lines = [line.strip() for line in montage_file.read_text(encoding="utf-8").splitlines()]
        positions_start = lines.index("Positions") + 1
        labels_start = lines.index("Labels") + 1
        positions = [[float(value) / 1000 for value in line.split()] for line in lines[positions_start:labels_start - 1]]
        labels = lines[labels_start:labels_start + len(positions)]
        coordinate_map = dict(zip(labels, positions))
        missing = set(channel_names) - coordinate_map.keys()
        if missing:
            raise ValueError(f"Electrodes missing from standard_1005 montage: {sorted(missing)}")
        coordinates = torch.tensor([coordinate_map[name] for name in channel_names], dtype=torch.float32)
    if coordinates.shape != (len(channel_names), 3) or not bool(torch.isfinite(coordinates).all()):
        raise ValueError("Electrode coordinates must have finite shape [n_channels, 3] in metres.")
    return coordinates


def build_knn_adjacency(coordinates: Tensor, k: int = 4) -> Tensor:
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("KNN coordinates must be [electrodes, 3].")
    nodes = coordinates.shape[0]
    if not 0 <= k < nodes:
        raise ValueError("KNN k must be between zero and n_channels - 1.")
    distances = torch.cdist(coordinates, coordinates)
    distances.fill_diagonal_(float("inf"))
    neighbors = distances.topk(k, largest=False).indices
    adjacency = torch.eye(nodes, dtype=torch.bool, device=coordinates.device)
    adjacency.scatter_(1, neighbors, True)
    return adjacency | adjacency.T


class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, adjacency: Tensor, config: Mapping, dropout: float):
        super().__init__()
        if heads < 1 or d_model % heads:
            raise ValueError("GAT d_model must be divisible by gat_heads.")
        self.heads = heads
        self.head_dim = d_model // heads
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.attention_source = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attention_target = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)
        self.register_buffer("adjacency", adjacency.clone())
        self.negative_slope = config.get("gat_negative_slope", 0.2)
        self.attention_dropout = nn.Dropout(config.get("gat_attention_dropout", dropout))
        self.output_dropout = nn.Dropout(dropout)
        self.residual = config.get("gat_residual", True)
        normalization = config.get("gat_normalization", "layernorm")
        if normalization not in {"layernorm", "none"}:
            raise ValueError("GAT normalization must be layernorm or none.")
        self.norm = nn.LayerNorm(d_model) if normalization == "layernorm" else nn.Identity()

    def forward(self, x: Tensor, return_attention: bool = False):
        batch_size, nodes, d_model = x.shape
        projected = self.projection(x).reshape(batch_size, nodes, self.heads, self.head_dim).transpose(1, 2)
        source_scores = (projected * self.attention_source[None, :, None]).sum(dim=-1)
        target_scores = (projected * self.attention_target[None, :, None]).sum(dim=-1)
        scores = F.leaky_relu(source_scores.unsqueeze(-1) + target_scores.unsqueeze(-2), self.negative_slope)
        scores = scores.masked_fill(~self.adjacency[None, None], float("-inf"))
        attention = scores.softmax(dim=-1)
        messages = self.attention_dropout(attention) @ projected
        messages = messages.transpose(1, 2).reshape(batch_size, nodes, d_model)
        output = self.output_dropout(F.gelu(messages))
        if self.residual:
            output = x + output
        output = self.norm(output)
        if return_attention:
            return output, attention
        return output


class SpatialPriorGAT(nn.Module):
    def __init__(self, coordinates: Tensor, d_model: int = 64, config: Optional[Mapping] = None, dropout: float = 0.3):
        super().__init__()
        config = config or {"knn_k": 4, "gat_layers": 2, "gat_heads": 4}
        if config["gat_layers"] < 1:
            raise ValueError("GAT requires at least one layer.")
        self.register_buffer("coordinates", coordinates.clone())
        self.register_buffer("adjacency", build_knn_adjacency(coordinates, config["knn_k"]))
        self.layers = nn.ModuleList([
            GraphAttentionLayer(d_model, config["gat_heads"], self.adjacency, config, dropout)
            for _ in range(config["gat_layers"])
        ])

    def forward(self, x: Tensor, return_attention: bool = False):
        if not return_attention:
            for layer in self.layers:
                x = layer(x)
            return x
        attentions = []
        for layer in self.layers:
            x, attention = layer(x, return_attention=True)
            attentions.append(attention)
        return x, torch.stack(attentions, dim=1)


class AdaptiveROI(nn.Module):
    def __init__(self, coordinates: Tensor, channel_names: Sequence[str], d_model: int = 64, config: Optional[Mapping] = None):
        super().__init__()
        config = config or {
            "n_regions": 3, "region_anchor_names": ["C3", "Cz", "C4"],
            "region_sigma_m": [0.06, 0.06, 0.06], "prior_strength": 1.0, "roi_eps": 1e-6,
        }
        anchors = config["region_anchor_names"]
        n_regions = config["n_regions"]
        if len(anchors) != n_regions or n_regions != 3:
            raise ValueError("This first-version model requires three configured ROI anchors.")
        if any(name not in channel_names for name in anchors):
            raise ValueError("ROI anchors must occur in data.channel_names.")
        sigma = torch.as_tensor(config["region_sigma_m"], dtype=coordinates.dtype)
        if sigma.shape != (n_regions,) or not bool((sigma > 0).all()):
            raise ValueError("region_sigma_m must contain three positive bandwidths in metres.")
        anchor_positions = coordinates[[channel_names.index(name) for name in anchors]]
        squared_distance = (coordinates[:, None] - anchor_positions[None, :]).square().sum(dim=-1)
        prior = (-squared_distance / (2 * sigma.square())).softmax(dim=-1)
        self.register_buffer("spatial_prior", prior)
        self.eps = float(config["roi_eps"])
        self.prior_strength = float(config["prior_strength"])
        if self.eps <= 0:
            raise ValueError("ROI epsilon must be positive.")
        self.feature_projection = nn.Linear(d_model, d_model)
        self.queries = nn.Parameter(torch.empty(n_regions, d_model))
        nn.init.normal_(self.queries, std=1 / math.sqrt(d_model))
        self.scale = math.sqrt(d_model)

    def forward(self, x: Tensor, return_assignment: bool = False):
        scores = self.feature_projection(x) @ self.queries.T / self.scale
        scores = scores + self.prior_strength * (self.spatial_prior + self.eps).log()
        assignment = scores.softmax(dim=-1)
        numerator = assignment.transpose(1, 2) @ x
        denominator = assignment.sum(dim=1).unsqueeze(-1) + self.eps
        regions = numerator / denominator
        if return_assignment:
            return regions, assignment
        return regions


class RegionTransformer(nn.Module):
    def __init__(self, d_model: int, config: Mapping, dropout: float = 0.3):
        super().__init__()
        self.encoder = _transformer(d_model, config, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class CrossBlock(nn.Module):
    def __init__(self, d_model: int = 64, heads: int = 4, dim_feedforward: int = 128, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query: Tensor, context: Tensor, return_attention: bool = False):
        update, weights = self.attention(query, context, context, need_weights=return_attention, average_attn_weights=False)
        output = self.norm1(query + self.attention_dropout(update))
        output = self.norm2(output + self.ffn_dropout(self.ffn(output)))
        if return_attention:
            return output, weights
        return output


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, d_model: int = 64, heads: int = 4, dim_feedforward: int = 128, dropout: float = 0.3):
        super().__init__()
        self.time_to_region = CrossBlock(d_model, heads, dim_feedforward, dropout)
        self.region_to_time = CrossBlock(d_model, heads, dim_feedforward, dropout)

    def forward(self, time_tokens: Tensor, region_tokens: Tensor, return_attention: bool = False):
        # Both directions read the original tokens, with independent parameters.
        time_result = self.time_to_region(time_tokens, region_tokens, return_attention)
        region_result = self.region_to_time(region_tokens, time_tokens, return_attention)
        if return_attention:
            time_output, time_weights = time_result
            region_output, region_weights = region_result
            return time_output, region_output, time_weights, region_weights
        return time_result, region_result


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.query, std=1 / math.sqrt(d_model))
        self.scale = math.sqrt(d_model)

    def forward(self, x: Tensor, return_weights: bool = False):
        weights = (x @ self.query / self.scale).softmax(dim=-1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        if return_weights:
            return pooled, weights
        return pooled


class AdaptiveFusion(nn.Module):
    def __init__(self, d_model: int = 64, dropout: float = 0.3):
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)
        self.projection = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, time_features: Tensor, region_features: Tensor, return_gate: bool = False):
        gate = self.gate(torch.cat((time_features, region_features), dim=-1)).sigmoid()
        weighted = torch.cat((gate * time_features, (1 - gate) * region_features), dim=-1)
        fused = self.dropout(self.norm(F.gelu(self.projection(weighted))))
        if return_gate:
            return fused, gate
        return fused


class DualBranchMINet(nn.Module):
    """Dual frequency-time/spatial branches with parallel cross-attention."""

    def __init__(self, config: Mapping):
        super().__init__()
        data_config = config["data"]
        model_config = config["model"]
        d_model = model_config["d_model"]
        dropout = model_config["dropout"]
        self.n_channels = data_config["n_channels"]
        self.target_samples = round((data_config["epoch_end_sec"] - data_config["epoch_start_sec"]) * data_config["sampling_rate"])
        if len(data_config["channel_names"]) != self.n_channels or len(set(data_config["channel_names"])) != self.n_channels:
            raise ValueError("data.channel_names must contain n_channels unique names.")
        if d_model < 1 or self.target_samples < 1:
            raise ValueError("d_model and target epoch length must be positive.")
        spatial_config = model_config["spatial"]
        coordinates = load_electrode_coordinates(data_config["channel_names"], spatial_config)
        self.register_buffer("electrode_coordinates", coordinates)
        self.frequency_branch = FrequencyBranch(config)
        self.electrode_encoder = SharedElectrodeEncoder(d_model, spatial_config)
        self.gat = SpatialPriorGAT(coordinates, d_model, spatial_config, dropout)
        self.roi = AdaptiveROI(coordinates, data_config["channel_names"], d_model, spatial_config)
        self.region_transformer = RegionTransformer(d_model, model_config["region_transformer"], dropout)
        self.cross_attention = BidirectionalCrossAttention(d_model=d_model, dropout=dropout, **model_config["cross_attention"])
        self.time_pool = AttentionPooling(d_model)
        self.region_pool = AttentionPooling(d_model)
        self.fusion = AdaptiveFusion(d_model, model_config["fusion"]["dropout"])
        classifier_config = model_config["classifier"]
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_config["hidden_dim"]), nn.GELU(),
            nn.Dropout(classifier_config["dropout"]), nn.Linear(classifier_config["hidden_dim"], model_config["num_classes"]),
        )

    def forward(self, x_spatial: Tensor, x_frequency: Tensor, return_aux: bool = False):
        if x_spatial.ndim != 3 or x_spatial.shape[1:] != (self.n_channels, self.target_samples):
            raise ValueError(f"Spatial input must be [batch, {self.n_channels}, {self.target_samples}].")
        if x_frequency.ndim != 3 or x_frequency.shape[:2] != x_spatial.shape[:2]:
            raise ValueError("Frequency and spatial inputs must have matching batch and electrode dimensions.")
        frequency_result = self.frequency_branch(x_frequency, return_aux=return_aux)
        electrode_features = self.electrode_encoder(x_spatial)
        graph_result = self.gat(electrode_features, return_attention=return_aux)
        if return_aux:
            time_tokens, aux = frequency_result
            graph_features, gat_attention = graph_result
        else:
            time_tokens = frequency_result
            graph_features = graph_result
        region_result = self.roi(graph_features, return_assignment=return_aux)
        if return_aux:
            raw_region_tokens, roi_assignment = region_result
        else:
            raw_region_tokens = region_result
        region_tokens = self.region_transformer(raw_region_tokens)
        cross_result = self.cross_attention(time_tokens, region_tokens, return_attention=return_aux)
        if return_aux:
            cross_time_tokens, cross_region_tokens, time_to_region, region_to_time = cross_result
        else:
            cross_time_tokens, cross_region_tokens = cross_result
        pooled_time, time_pool_weights = self.time_pool(cross_time_tokens, return_weights=True)
        pooled_region, region_pool_weights = self.region_pool(cross_region_tokens, return_weights=True)
        fused_features, fusion_gate = self.fusion(pooled_time, pooled_region, return_gate=True)
        logits = self.classifier(fused_features)
        if return_aux:
            aux.update({
                "electrode_features": electrode_features,
                "graph_features": graph_features,
                "gat_attention": gat_attention,
                "roi_assignment": roi_assignment,
                "raw_region_tokens": raw_region_tokens,
                "region_tokens": region_tokens,
                "cross_time_tokens": cross_time_tokens,
                "cross_region_tokens": cross_region_tokens,
                "cross_attention_time_to_region": time_to_region,
                "cross_attention_region_to_time": region_to_time,
                "pooled_time": pooled_time,
                "pooled_region": pooled_region,
                "time_pool_weights": time_pool_weights,
                "region_pool_weights": region_pool_weights,
                "fusion_gate": fusion_gate,
                "fused_features": fused_features,
            })
            return logits, aux
        return logits
