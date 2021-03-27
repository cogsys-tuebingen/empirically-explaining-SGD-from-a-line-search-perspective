# This script calculates the distance MAE between full-batch losses along lines  (Figure 3) and the MAE of the parabolic approximation (Figure 4)


from line_statistics import *

data_load_path = "CIFAR10_mom0_resnet20_augment"

np.random.seed(42)

save_path, data_save_path, line_plot_save_path, statistics_plot_save_path, mean_line_save_path, _ = create_save_dirs(
    data_load_path)
data_files = load_and_sort_data_files(data_load_path)
if len(os.listdir(mean_line_save_path)) == 0:
    calculate_and_save_line_means(data_load_path, data_files, mean_line_save_path)

max_degree = 4
window = 0.2  # for sgd
pgf_height_and_width = "width=10.5cm,height=7.2cm"
for degree in range(max_degree):
    plot_polynomial_fit_error_over_lines(-window, window, degree + 1, mean_line_save_path, statistics_plot_save_path,
                                         scale_factor=10, height_and_width=pgf_height_and_width)
compare_mean_lines(-window, window, mean_line_save_path, statistics_plot_save_path, scale_factor=10,
                   height_and_width=pgf_height_and_width)
