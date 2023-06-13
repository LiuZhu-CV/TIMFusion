from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat normal_tv normal_tv_concat chain_model chain_model_concat chain_fusion chain_fusion_concat')


PRIMITIVES = [
'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'eca_attention',
    'se_attention',
    'ResidualModules',
    'EnhanceREsidualModules'
]


Genotype2 = namedtuple('Genotype', 'normal normal_concat')

fusion_ir = Genotype2(normal=[('SepConv_3_1', 0), ('SPAattention_3', 1)],normal_concat=None)
fusion_vis = Genotype2(normal=[('Residualblocks_3_1', 0), ('SPAattention_3', 1)],normal_concat=None)


fusion_M = Genotype(normal=[('Denseblocks_3_1', 0), ('DilConv_3_2', 1)], normal_concat=[1, 2, 3],
                        normal_tv=[('Residualblocks_3_1', 0), ('Residualblocks_3_2', 1)], normal_tv_concat=[1, 2, 3],
                        chain_model=[('Residualblocks_3_1', 0), ('Residualblocks_3_1', 1)], chain_model_concat=[1, 2, 3],
                        chain_fusion=[('Residualblocks_3_1', 0), ('DilConv_3_2', 1)], chain_fusion_concat=[1, 2, 3])
fusion_mri = Genotype2(normal=[('Residualblocks_5_1', 0), ('Residualblocks_5_1', 1)],normal_concat=None)
fusion_ct = Genotype2(normal=[('Residualblocks_5_1', 0), ('SPAattention_3', 1)],normal_concat=None)
fusion_mri_pet = Genotype2(normal=[('SepConv_3_1', 0), ('Residualblocks_5_1', 1)],normal_concat=None)
fusion_pet = Genotype2(normal=[('Residualblocks_3_1', 0), ('Residualblocks_5_1', 1)],normal_concat=None)
fusion_mri_spect = Genotype2(normal=[('Residualblocks_5_1', 0), ('Denseblocks_3_1', 1)],normal_concat=None)
fusion_spect = Genotype2(normal=[('Residualblocks_3_1', 0), ('SPAattention_3', 1)],normal_concat=None)

## new architecture
fusion_ir_proposed = Genotype2(normal=[('SPAattention_3', 0), ('DilConv_3_2', 1)],normal_concat=None)
fusion_vis_proposed = Genotype2(normal=[('ECAattention_3', 0), ('SPAattention_3', 1)],normal_concat=None)

fusion_ir_Darts = Genotype2(normal=[('ECAattention_3', 0), ('ECAattention_3', 1)],normal_concat=None)
fusion_vis_Darts = Genotype2(normal=[('SPAattention_3', 0), ('SPAattention_3', 1)],normal_concat=None)

fusion_light = Genotype(normal=[('Residualblocks_3_1', 0), ('ECAattention_3', 1)], normal_concat=[1, 2, 3],
                        normal_tv=[('Denseblocks_5_1', 0), ('Denseblocks_5_1', 1)], normal_tv_concat=[1, 2, 3],
                        chain_model=[('Denseblocks_5_1', 0), ('Denseblocks_5_1', 1)], chain_model_concat=[1, 2, 3],
                        chain_fusion=[('Residualblocks_3_1', 0), ('Denseblocks_3_1', 1)], chain_fusion_concat=[1, 2, 3])
#########Jointly Searching
fusion_M4 = Genotype(normal=[('SPAattention_3', 0), ('DilConv_3_2', 1)], normal_concat=[1, 2, 3],
                        normal_tv=[('Denseblocks_5_1', 0), ('Denseblocks_5_1', 1)], normal_tv_concat=[1, 2, 3],
                        chain_model=[('Denseblocks_5_1', 0), ('Denseblocks_5_1', 1)], chain_model_concat=[1, 2, 3],
                        chain_fusion=[('SPAattention_3', 0), ('Residualblocks_5_1', 1)], chain_fusion_concat=[1, 2, 3])
fusion_ir_proposed2 = Genotype2(normal=[('Residualblocks_3_1', 0), ('SPAattention_3', 1)],normal_concat=None)
fusion_vis_proposed2 = Genotype2(normal=[('ECAattention_3', 0), ('Residualblocks_3_1', 1)],normal_concat=None)