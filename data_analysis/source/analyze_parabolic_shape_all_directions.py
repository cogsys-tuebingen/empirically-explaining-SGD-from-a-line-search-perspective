# This script calculates the distance MAE between full-batch losses along lines  (Figure 3) and the MAE of the parabolic approximation (Figure 4)
# In contrast to "analyze_parabolic_shape.py"  this script compares sets of resulting lines of multiple noisy gradient directions originating from the same position.
from line_statistics import *

data_load_path = "CIFAR10_mom0_resnet20_augment_on_pos"
np.random.seed(42)

data_files = load_and_sort_data_files(data_load_path)
position_groups = create_and_sort_position_groups(data_files)
for position_group in position_groups:
    name = "group_" + position_group[0].split("_")[2]
    save_path, data_save_path, line_plot_save_path, statistics_plot_save_path, mean_line_save_path, _ = create_save_dirs(
        data_load_path, name)
    if len(os.listdir(mean_line_save_path)) == 0:
        calculate_and_save_line_means(data_load_path, position_group, mean_line_save_path)

    plot_distance_to_mean_min_over_lines(mean_line_save_path, statistics_plot_save_path, name)
    max_degree = 5
    window = 0.2
    for degree in range(max_degree):
        plot_polynomial_fit_error_over_lines(-window, window, degree + 1, mean_line_save_path,
                                             statistics_plot_save_path, name)
    compare_mean_lines(-window, window, mean_line_save_path, statistics_plot_save_path, name, scip_first_50=True,
                       scale_factor=1)
