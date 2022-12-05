from analyse_specific import performe_visual_analysis

# Setup
#####################################################
###to-do: set correct path

###to-do: set correct class2name

###to-do: input list of testset names in order of testing. Put all sets here and sort out the heavy computational ones in a later block. Order is important!
# ls_testsets = ["val","iid","MSKCC"]
ls_testsets = ["val", "iid", "brighter_1"]
# ls_testsets = ["val", "iid", "subclass"]


###to-do: set correct class to plot
class2plot = [0, 1]

###to:do: set correct test sets to plot
# test_datasets = ["iid", "subclass"]  # must be values of ls_testsets!
test_datasets = ["iid", "brighter_1"]  # must be values of ls_testsets!

#####################################################


# path ="/home/l049e/cluster_checkpoints/dermoscopyall_os_0_ep_30/ce_lr3e-05_CosineAnnealingWarmRestarts"
# path= "/home/l049e/cluster_checkpoints/xray_chestall_lr_tune/ce_lr0.00025"
# path ="/home/l049e/cluster_checkpoints/2022_08_16/dermoscopyall_run1/ef_troubleshootdevries"
# path = "/home/l049e/cluster_checkpoints/xray_chestallbutmimic_ce_run_1/ce"
# path = "/home/l049e/cluster_checkpoints/ham10000multi_ce_run1/ce"
# path = "/home/l049e/cluster_checkpoints/xray_chestall_run_1/cross_entropy_mcd_ep_30"
# path = "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_lidc_idriall_texture_run_1/confidnet_mcd"

# path = "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_lidc_idriall_run_1/confidnet_mcd"
# path = "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_dermoscopyallham10000subclass_run_1/confidnet_mcd"
# path = "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_dermoscopyallbutmskcc_run_1/confidnet_mcd"

# path = "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_xray_chestall_run_1/confidnet_mcd"
# path = "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_xray_chestallbutnih14_run_1/confidnet_mcd"
# path = "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_xray_chestallbutchexpert_run_1/confidnet_mcd"
# path = "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_rxrx1all_large_set2_run_1/confidnet_mcd"

path = "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_dermoscopyall_run_1/confidnet_mcd"
# path = "/home/l049e/cluster_checkpoints/dermoscopyall_ce_run1/ce"

class2name = OrderedDict({0: "benign", 1: "malignant"})
# class2name = OrderedDict({0:"No_Finding",1:"Cardiomegaly",2:"Edema",3:"Consolidation",4:"Pneumonia",5:"Atelectasis",6:"Pneumothorax",7:"Plaural Effusion"})
# class2name = OrderedDict({0:"akiec",1:"bcc",2:"bkl",3:"df",4:"mel",5:"nv",6:"vasc"})
# class2name_helper = {}
# for i in range(1139):
#    class2name_helper[i] = i
# class2name = OrderedDict(class2name_helper)

path_list = [
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_dermoscopyall_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_dermoscopyall_run_1/devries_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_dermoscopyallbutmskcc_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_dermoscopyallham10000subclass_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_lidc_idriall_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_lidc_idriall_calcification_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_lidc_idriall_texture_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_xray_chestall_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_xray_chestallbutnih14_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_xray_chestallbutchexpert_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_rxrx1all_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_rxrx1all_large_set1_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_rxrx1all_large_set2_run_1/confidnet_mcd",
]
iid_paths = [
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_dermoscopyall_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_xray_chestall_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_lidc_idriall_run_1/confidnet_mcd",
    "/home/l049e/E130-Personal/Kobelke/cluster_checkpoints/ms_rxrx1all_run_1/confidnet_mcd",
]

for path in path_list[1:2]:
    if "dermoscopy" in path:
        class2plot = [0, 1]
        class2name = OrderedDict({0: "benign", 1: "malignant"})
        domain = "dermoscopy"
    elif "lidc" in path:
        class2plot = [1]
        class2name = OrderedDict({0: "benign", 1: "malignant"})
        domain = "lidc"
    elif "xray" in path:
        class2name = OrderedDict(
            {
                0: "No_Finding",
                1: "Cardiomegaly",
                2: "Edema",
                3: "Consolidation",
                4: "Pneumonia",
                5: "Atelectasis",
                6: "Pneumothorax",
                7: "Plaural Effusion",
            }
        )
        class2plot = [7]

        domain = "chest"
    elif "rxrx1" in path:
        class2name_helper = {}
        class2plot = [100]

        for i in range(1139):
            class2name_helper[i] = i
        class2name = OrderedDict(class2name_helper)
        domain = "rxrx1"
    if path in iid_paths:
        ls_testsets = ["val", "iid", "brighter_1"]
        test_datasets = ["iid", "brighter_1"]
    else:
        ls_testsets = ["val", "iid", "subclass"]
        test_datasets = ["iid", "subclass"]
    performe_visual_analysis(
        path=path,
        class2name=class2name,
        class2plot=class2plot,
        ls_testsets=ls_testsets,
        test_datasets=test_datasets,
        domain=domain,
    )
