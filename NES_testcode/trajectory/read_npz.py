import numpy as np

data = np.load('./scen1-new/traj_sce1_arch_fromLobbyto801_show_at_10min_40ms.npz', allow_pickle=True)
print(data['UE_posX_at_t'].shape)