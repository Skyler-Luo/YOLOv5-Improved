import sys
from pathlib import Path
import torch.nn as nn

class DependencyMissing(object):
    def __init__(self, module_name, dependencies):
        self.module_name = module_name
        self.dependencies = dependencies

    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"Attention module '{self.module_name}' requires package(s): {self.dependencies}. "
            f"Please install via: pip install {self.dependencies}"
        )

# Direct imports from local files
from .A2Attention import DoubleAttention
from .BAM import BAMBlock
from .CAA import CAA
from .CBAM import CBAMBlock
from .CPCA import CPCA
from .CoTAttention import CoTAttention
from .CoordAttention import CoordAtt
from .ECA import EfficientChannelAttention
from .ELA import ELA
from .EMA import EMA
from .GAM import GAM_Attention
from .LSKA import LSKA
from .LSKBlock import LSKblock
from .MHSA import MHSA
from .MLCA import MLCA
from .ParNetAttention import ParNetAttention
from .PolarizedSelfAttention import ParallelPolarizedSelfAttention
from .S2Attention import S2Attention
from .SE import SEAttention
from .SGE import SpatialGroupEnhance
from .SK import SKAttention
from .SequentialSelfAttention import SequentialPolarizedSelfAttention
from .ShuffleAttention import ShuffleAttention
from .SimAM import SimAM
from .TripletAttention import TripletAttention

# Conditional imports with extra dependencies
try:
    from .Biformer import BiLevelRoutingAttention
except ImportError:
    BiLevelRoutingAttention = DependencyMissing('BiLevelRoutingAttention', 'einops')

try:
    from .CloAttention import EfficientAttention
except ImportError:
    EfficientAttention = DependencyMissing('EfficientAttention', 'efficientnet_pytorch')

try:
    from .DAttention import DAttention
except ImportError:
    DAttention = DependencyMissing('DAttention', 'timm einops')

try:
    from .EffectiveSE import EffectiveSEModule
except ImportError:
    EffectiveSEModule = DependencyMissing('EffectiveSEModule', 'timm')

try:
    from .GC import GlobalContext
except ImportError:
    GlobalContext = DependencyMissing('GlobalContext', 'timm')

try:
    from .GE import GatherExcite
except ImportError:
    GatherExcite = DependencyMissing('GatherExcite', 'timm')

try:
    from .MobileViTAttention import MobileViTAttention
except ImportError:
    MobileViTAttention = DependencyMissing('MobileViTAttention', 'einops')

__all__ = [
    'DoubleAttention', 'BAMBlock', 'BiLevelRoutingAttention', 'CAA', 'CBAMBlock',
    'CPCA', 'EfficientAttention', 'CoTAttention', 'CoordAtt', 'DAttention',
    'EfficientChannelAttention', 'ELA', 'EMA', 'EffectiveSEModule', 'GAM_Attention',
    'GlobalContext', 'GatherExcite', 'LSKA', 'LSKblock', 'MHSA', 'MLCA',
    'MobileViTAttention', 'ParNetAttention', 'ParallelPolarizedSelfAttention',
    'S2Attention', 'SEAttention', 'SpatialGroupEnhance', 'SKAttention',
    'SequentialPolarizedSelfAttention', 'ShuffleAttention', 'SimAM', 'TripletAttention'
]
