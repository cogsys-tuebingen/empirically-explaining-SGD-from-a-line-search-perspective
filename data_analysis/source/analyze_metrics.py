
# This


from line_statistics import *

# Create plots of the full-batch loss along lines such as in Figure 
is_plot = False
plot_interval=10

data_load_path = "CIFAR10_mom00_resnet20_augment"
is_momentum=True

pgf_height_and_width="width=10.5cm,height=8cm"
experiments = {
    "bs": [32,64,128,256,512,1024,2048],
    "bs_ori": [128],
    "sgd_lrs": [[0.05]],
    'pal_mus': [[0.1]],
    'ri': [100],
    'c_apal': [0.9],
}

np.random.seed(42)

config_dicts = list(dict_product(experiments))

new_config_dict_list=[]
data_list=[]
save_path, data_save_path, line_plot_save_path, statistics_plot_save_path, _,batch_compare_path = create_save_dirs(data_load_path)
for i, config_dict in enumerate(config_dicts):
    print(config_dict)

    data_files = load_and_sort_data_files(data_load_path)

    if not check_if_line_statistics_data_exist(config_dict, data_save_path) or is_plot:
        data_dict,new_config_dict = get_pal_sgd_bls_statistics_with_pal_adaption(data_load_path, data_files, config_dict, data_save_path, is_plot,[-0.5,0.5],
                                                       line_plot_save_path,plot_interval,pgf_height_and_width,is_momentum)

    else:
        data_dict, new_config_dict= read_line_statistics_data(config_dict, data_save_path)
        new_config_dict_list.append(new_config_dict)
    data_list.append(data_dict)

    print("evaluation done, start plotting")
    plot_sgd_pal_data_dict(data_dict, statistics_plot_save_path, new_config_dict,pgf_height_and_width)

if len(config_dicts)>1:
    plot_directional_derivatives(data_list,new_config_dict_list, batch_compare_path,pgf_height_and_width)
    plot_directional_derivatives_ratios(data_list, new_config_dict_list, batch_compare_path,pgf_height_and_width)
    #plot_directional_curvatures(data_list, new_config_dict_list, statistics_plot_save_path,pgf_height_and_width)
    #plot_directional_curvature_ratios(data_list, new_config_dict_list, statistics_plot_save_path,pgf_height_and_width)
    plot_fb_step_to_min_to_db_grad_ratio_batches(data_list, new_config_dict_list, batch_compare_path)
    #plot_variances(data_list,new_config_dict_list, statistics_plot_save_path)
    plot_improvement_over_batch_sizes(data_list,new_config_dict_list, batch_compare_path,pgf_height_and_width,accumulate=True,)
    plot_distance_over_batch_sizes(data_list,new_config_dict_list, batch_compare_path,pgf_height_and_width,accumulate=True)
    plot_improvement_over_batch_sizes(data_list,new_config_dict_list, batch_compare_path,pgf_height_and_width,accumulate=False)
    plot_distance_over_batch_sizes(data_list,new_config_dict_list, batch_compare_path,pgf_height_and_width,accumulate=False)