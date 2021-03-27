from util import *


def calculate_and_save_line_means(data_load_path, data_files, mean_line_save_path):
    for step, line_data_file in enumerate(data_files):
        train_line_losses, val_line_losses, sample_positions, direction_batch_number, sgd_update_step, direction_norm, is_extrapolation, direction_batch_number2 = load_line_data_no_dir_batch_data(
            data_load_path,
            line_data_file)

        mean_line_train_losses = np.mean(train_line_losses, axis=0)

        # min analysis
        mean_line_train_losses_min_data = get_value_index_location_and_error_of_min(mean_line_train_losses,
                                                                                    sample_positions)
        min_position = mean_line_train_losses_min_data["min_position"]
        min_value = mean_line_train_losses_min_data["loss_value"]
        file_path = os.path.join(mean_line_save_path, line_data_file)
        write_pickle((mean_line_train_losses - min_value, sample_positions, min_position, min_value), file_path)
        print(step)


def plot_distance_to_mean_min_over_lines(mean_line_save_path, statistics_plot_save_path, name=""):
    mean_files = load_and_sort_data_files(mean_line_save_path)
    min_positions = []
    for step, file in enumerate(mean_files):
        with open(os.path.join(mean_line_save_path, file), 'rb') as f:
            _, _, min_position, _ = np.load(f, allow_pickle=True)
            min_positions.append(min_position)

    fig = plt.figure()
    plt.title("Distance to Minimum " + name)
    plt.plot(range(len(min_positions)), min_positions, linewidth=1.0)
    plt.xlabel("line number")
    plt.yscale("log")
    plt.savefig("{0}/{1}.png".format(statistics_plot_save_path, "distances_to_minimum"), dpi=200)
    plt.close()


def plot_polynomial_fit_error_over_lines(from_, to, degree, mean_line_save_path, statistics_plot_save_path, name="",
                                         scale_factor=10, height_and_width=""):
    mean_files = load_and_sort_data_files(mean_line_save_path)
    fit_errors = []
    coefficients_list = []
    for step, file in enumerate(mean_files):
        with open(os.path.join(mean_line_save_path, file), 'rb') as f:
            centered_mean_line_train_losses, sample_positions, min_position, min_value = np.load(f, allow_pickle=True)
            indexes_in_measure_interval = \
                np.where(np.logical_and(sample_positions >= 0 + from_, sample_positions <= 0 + to))[
                    0]
            centered_mean_line_train_losses = centered_mean_line_train_losses[indexes_in_measure_interval]
            mean_line_train_losses = centered_mean_line_train_losses + min_value
            sample_positions = sample_positions[indexes_in_measure_interval]
            x = list(sample_positions.flatten())
            y = list(mean_line_train_losses.flatten())
            coefficients = np.polyfit(x, y, degree)
            coefficients_list.append(np.flip(coefficients))
            fit_error = get_mean_absolut_error(x, mean_line_train_losses,
                                               coefficients)
            fit_errors.append(fit_error)
    fit_errors = fit_errors[::scale_factor]
    coefficients_list = coefficients_list[::scale_factor]
    fig = plt.figure()
    plt.title("MAE of fitted degree " + str(degree) + " polynomials")
    plt.plot(np.array(range(len(fit_errors))) * scale_factor, fit_errors)
    print(degree)
    print(np.mean(fit_errors))
    plt.xlabel("line number")
    plt.ylabel("MAE")
    # see pgfplot docu page 322
    # and https://tex.stackexchange.com/questions/31276/number-format-in-pgfplots-axis
    # change labes to be decimal not scientfic:
    config = "y tick label style={/pgf/number format/.cd,scaled y ticks = false,set thousands separator={},fixed,precision=4}"
    tikzplotlib.save("{0}/{1}_{2}.pgf".format(statistics_plot_save_path, "mae_of_polynomial_fit_of_degree", degree),
                     override_externals=True, strict=True, extra_axis_parameters=[height_and_width, config])
    plt.savefig("{0}/{1}.png".format(statistics_plot_save_path, "mae_of_polynomial_fit_of_degree" + str(degree)),
                dpi=200)

    plt.close()

    fig = plt.figure()

    label_dict = {}

    plt.title("Coefficients of  fitted degree " + str(degree) + " polynomials")
    for i in range(len(coefficients_list[0])):
        data = [d[i] for d in coefficients_list]
        data = data - np.min(data)
        label = "coef " + str(i)
        if i == 0:
            label = "c (offset)"
        if i == 1:
            label = "b (slope)"
        if i == 2:
            label = "a (curvature / 2)"

        plt.plot(np.array(range(len(data))) * scale_factor, data, label=label)

    plt.xlabel("line number")
    plt.ylabel("coefficient size")
    plt.legend()
    tikzplotlib.save(
        "{0}/{1}_{2}.pgf".format(statistics_plot_save_path, "coefficients_of_polynomial_fit_of_degree", degree,
                                 extra_axis_parameters=[height_and_width]),
        override_externals=True, extra_axis_parameters=[height_and_width])
    plt.savefig(
        "{0}/{1}.png".format(statistics_plot_save_path, "coefficients_of_polynomial_fit_of_degree" + str(degree)),
        dpi=200)

    plt.close()


def compare_mean_lines(from_, to, mean_line_save_path, statistics_plot_save_path, name="", scale_factor=10,
                       scip_first_50=False, height_and_width=""):
    mean_files = load_and_sort_data_files(mean_line_save_path)
    with open(os.path.join(mean_line_save_path, mean_files[0]), 'rb') as f:
        _, sample_positions, _, _ = np.load(f, allow_pickle=True)
    indexes_in_measure_interval = np.where(np.logical_and(sample_positions >= 0 + from_, sample_positions <= 0 + to))[
        0]
    mean_matrix = np.zeros((len(mean_files), len(indexes_in_measure_interval) - 2))
    for step, file in enumerate(mean_files):
        with open(os.path.join(mean_line_save_path, file), 'rb') as f:
            centered_mean_line_train_losses, sample_positions, min_position, min_value = np.load(f, allow_pickle=True)
            indexes_in_measure_interval = \
                np.where(np.logical_and(sample_positions > 0 + from_, sample_positions < 0 + to))[
                    0]
            centered_mean_line_train_losses = centered_mean_line_train_losses[indexes_in_measure_interval]
            mean_matrix[step, :] = centered_mean_line_train_losses[:(mean_matrix.shape[1])]

    similarity_matrix = np.zeros((mean_matrix.shape[0], mean_matrix.shape[0]))
    for line_num, line in enumerate(mean_matrix):
        sub_matrix = np.abs(mean_matrix - line)
        mae = np.sum(sub_matrix, axis=1) / len(line)
        similarity_matrix[line_num, :] = mae[:]

    relative_data_path_for_pgf = os.path.join(os.getcwd(), statistics_plot_save_path)
    down_scaling = scale_factor

    if "group_" in name:
        name = name.replace("group_", " line ")

    sim = similarity_matrix
    ori_max = np.max(sim)
    # sim = similarity_matrix[50: :down_scaling, 50::down_scaling]
    sim = similarity_matrix[::down_scaling, ::down_scaling]
    fig = plt.figure()
    plt.title("Distance Matrix" + name)
    cax = plt.gca().matshow(sim, vmin=0, vmax=ori_max)
    cbar = fig.colorbar(cax)
    cbar.set_label('MAE', rotation=270)

    plt.xlabel("line number")
    plt.ylabel("line number")
    tikzplotlib.save("{0}/{1}.pgf".format(statistics_plot_save_path, "distance_matrix"),
                     tex_relative_path_to_data=relative_data_path_for_pgf, override_externals=True,
                     extra_axis_parameters=[height_and_width])
    plt.savefig("{0}/{1}.png".format(statistics_plot_save_path, "distance_matrix"), dpi=200,
                bbox_inches='tight')

    plt.close()
    if not scip_first_50:
        sim = similarity_matrix[0:50, :50]
        fig = plt.figure()
        plt.title("Distance Matrix first 50 lines")
        cax = plt.gca().matshow(sim, vmin=0, vmax=np.max(sim))
        cbar = fig.colorbar(cax)
        cbar.set_label('MAE', rotation=270)
        plt.xlabel("line number")
        plt.ylabel("line number")
        tikzplotlib.save("{0}/{1}.pgf".format(statistics_plot_save_path, "distance_matrix_first_50"),
                         tex_relative_path_to_data=relative_data_path_for_pgf, override_externals=True)
        plt.savefig("{0}/{1}.png".format(statistics_plot_save_path, "distance_matrix_first_50"), dpi=200,
                    bbox_inches='tight')

        plt.close()
        sim = similarity_matrix[50:, 50:]
        ori_max = np.max(sim)
        sim = sim[::down_scaling, ::down_scaling]
        fig = plt.figure()
        plt.title("Distance Matrix after line 50 ")
        cax = plt.gca().matshow(sim, vmin=0, vmax=ori_max)
        cbar = fig.colorbar(cax)
        cbar.set_label('MAE', rotation=270)
        plt.xlabel("line number")
        plt.ylabel("line number")
        ax = plt.gca()
        labels = np.array([50, 2000, 4000, 6000, 8000])
        ax.set_xticks(labels // down_scaling)
        ax.set_xticklabels(labels)
        ax.set_yticks(labels // down_scaling)
        ax.set_yticklabels(labels)
        tikzplotlib.save("{0}/{1}.pgf".format(statistics_plot_save_path, "distance_matrix_after_line50"),
                         tex_relative_path_to_data=relative_data_path_for_pgf,
                         override_externals=True)  # dpi does unfortunatelly not work
        plt.savefig("{0}/{1}.png".format(statistics_plot_save_path, "distance_matrix_after_line50"), dpi=200,
                    bbox_inches='tight')

        # plt.show(block=True)
        plt.close()

    # consecutive distances:
    cosecutive_data = np.diagonal(similarity_matrix, offset=1)[::scale_factor]
    plt.title("Loss distance of consecutive lines")
    plt.plot(np.array(range(len(cosecutive_data))) * scale_factor, cosecutive_data)
    plt.ylabel("MAE")
    plt.xlabel("line number")
    plt.yscale("log")

    tikzplotlib.save("{0}/{1}.pgf".format(statistics_plot_save_path, "consecutive_line_distances"),
                     tex_relative_path_to_data=relative_data_path_for_pgf, override_externals=True,
                     extra_axis_parameters=[height_and_width])
    plt.savefig("{0}/{1}.png".format(statistics_plot_save_path, "consecutive_line_distances"), dpi=200,
                bbox_inches='tight')

    plt.close()


def get_direction_single_batch_losses(batch_size, ori_batch_size, train_line_losses_uncombinded, direction_batch_losses,
                                      sample_postions):
    if batch_size == ori_batch_size:
        return direction_batch_losses
    elif batch_size < ori_batch_size:
        sorted_losses = sort_losses_by_directional_derivative(direction_batch_losses, sample_postions)
        losses = sorted_losses[:batch_size,
                 :]  # Todo on coould do this better by estimating the mean gradient value shift between the train losses and the direction batch losses.
        return losses
    elif batch_size > ori_batch_size:
        indexes = np.random.choice(train_line_losses_uncombinded.shape[0], batch_size - ori_batch_size, replace=False)
        non_ori_batch_losses = train_line_losses_uncombinded[indexes, :]
        losses = np.vstack((non_ori_batch_losses, direction_batch_losses))
        return losses


def sort_losses_by_directional_derivative(losses, sample_positions):
    zero_index = np.where(sample_positions == 0)[0][0]
    deriv_losses = losses[:, zero_index - 1:zero_index + 2]
    deriv_positions = sample_positions[zero_index - 1:zero_index + 2]
    g = np.gradient(deriv_losses, deriv_positions, axis=1)[:, 1]
    sort_indexes = g.argsort()
    sorted_losses = losses.copy()[sort_indexes, :]

    return sorted_losses


def get_pal_sgd_bls_statistics_with_pal_adaption(data_load_path, data_files, config_dict, data_save_path, is_plot,
                                                 x_plot_interval,
                                                 plot_save_path, figure_interval, height_and_width, is_momentum):
    batch_size = config_dict["bs"]
    ori_batch_size = config_dict["bs_ori"]
    sgd_learning_rates = config_dict["sgd_lrs"]
    pal_measuring_step_sizes = config_dict["pal_mus"]
    apal_resample_interval = config_dict["ri"]

    mean_min_locations = []
    direction_batch_min_locations = []
    sgd_step_locations = [[] for x in range(len(sgd_learning_rates))]
    pal_step_locations = [[] for x in pal_measuring_step_sizes]
    fb_pal_step_locations = [[] for x in pal_measuring_step_sizes]
    apal_step_locations = [[] for x in pal_measuring_step_sizes]
    apal_unsmoothed_step_locations = [[] for x in pal_measuring_step_sizes]
    apal_adapt_coefficients_list = [[1] for x in pal_measuring_step_sizes]
    original_sgd_locations = []

    distances_sgd_step_to_mean_min = [[] for x in range(len(sgd_learning_rates))]
    distances_pal_step_to_mean_min = [[] for x in pal_measuring_step_sizes]
    distances_fb_pal_step_to_mean_min = [[] for x in pal_measuring_step_sizes]
    distances_apal_step_to_mean_min = [[] for x in pal_measuring_step_sizes]
    distances_direction_batch_min_to_mean_min = []
    distances_original_sgd_step_to_mean_min = []

    improvements_sgd = [[] for x in range(len(sgd_learning_rates))]
    improvements_original_sgd = []
    improvements_pal = [[] for x in pal_measuring_step_sizes]
    improvements_fb_pal = [[] for x in pal_measuring_step_sizes]
    improvements_apal = [[] for x in pal_measuring_step_sizes]
    improvement_direction_batch_min = []
    improvement_real_loss_min = []

    directional_derivatives_direction_batch = []
    directional_curvature_directional_batch = []
    directional_derivatives_mean_train_loss = []
    directional_curvature_mean_train_loss = []

    variances_at_0 = []
    sample_variances_at_0 = []

    directional_derivatives_direction_batch_momentum = 0

    for step, line_data_file in enumerate(data_files):
        if not is_plot or (is_plot and (step % figure_interval) == 0):
            train_line_losses_unstacked, val_line_losses, sample_positions, direction_batch_losses, sgd_update_step, direction_norm, is_extrapolation, direction_batch_number2 = load_line_data_no_dir_batch_data(
                data_load_path,
                line_data_file)

            train_line_losses = np.vstack((train_line_losses_unstacked, direction_batch_losses))

            batched_line_train_losses = batch_losses_at_each_line_location(train_line_losses, sample_positions,
                                                                           batch_size)
            batched_train_line_losses_unstacked = batch_losses_at_each_line_location(train_line_losses_unstacked,
                                                                                     sample_positions, batch_size)

            direction_single_batch_losses = get_direction_single_batch_losses(batch_size, ori_batch_size,
                                                                              train_line_losses_unstacked,
                                                                              direction_batch_losses, sample_positions)

            if ori_batch_size == batch_size:
                direction_batch_train_losses = np.mean(direction_batch_losses, axis=0)
            else:
                direction_batch_train_losses = np.mean(direction_single_batch_losses, axis=0)

            mean_line_train_losses = np.mean(batched_line_train_losses, axis=0)
            min_line_train_losses = np.min(batched_train_line_losses_unstacked, axis=0)
            max_line_train_losses = np.max(batched_train_line_losses_unstacked, axis=0)
            q1_line_train_losses = np.percentile(batched_line_train_losses, 25, axis=0)
            q2_line_train_losses = np.percentile(batched_line_train_losses, 50, axis=0)
            q3_line_train_losses = np.percentile(batched_line_train_losses, 75, axis=0)
            # min analysis
            mean_line_train_losses_min_data = get_value_index_location_and_error_of_min(mean_line_train_losses,
                                                                                        sample_positions)
            direction_batch_losses_min_data = get_value_index_location_and_error_of_min(direction_batch_train_losses,
                                                                                        sample_positions)
            ##############
            # directional derivatives
            ##############

            direction_batch_directional_derivative, direction_batch_directional_curvature = get_direcional_derivative(
                direction_batch_train_losses,
                sample_positions)
            if batch_size == ori_batch_size and not is_momentum:
                direction_batch_directional_derivative = - direction_norm

            directional_derivatives_direction_batch_momentum = directional_derivatives_direction_batch_momentum * 0.9 + direction_batch_directional_derivative
            if is_momentum:
                directional_derivatives_direction_batch.append(direction_batch_directional_derivative)
            else:
                directional_derivatives_direction_batch.append(directional_derivatives_direction_batch_momentum)
            directional_curvature_directional_batch.append(direction_batch_directional_curvature)
            mean_train_loss_directional_derivative, mean_train_loss_directional_curvature = get_direcional_derivative(
                mean_line_train_losses, sample_positions)
            directional_derivatives_mean_train_loss.append(mean_train_loss_directional_derivative)
            directional_curvature_mean_train_loss.append(mean_train_loss_directional_curvature)

            ##############
            # Variances
            ##############
            variance_at_0 = get_loss_variance_at(0, train_line_losses, sample_positions)
            variances_at_0.append(variance_at_0)

            sample_variance_at_0 = get_loss_variance_at(0, direction_single_batch_losses, sample_positions)
            sample_variances_at_0.append(sample_variance_at_0)

            ori_sgd_lr = sgd_update_step / direction_norm

            if step == 0:
                config_dict["ori_sgd_lr"] = np.round(ori_sgd_lr, 3)

            ##############
            # steps
            ##############

            mean_min_locations.append(mean_line_train_losses_min_data["min_position"])
            # direction batch step
            direction_batch_min_locations.append(direction_batch_losses_min_data["min_position"])

            # original sgd
            if batch_size == ori_batch_size:
                original_sgd_step = sgd_update_step
            else:
                original_sgd_step = ori_sgd_lr * -direction_batch_directional_derivative
            original_sgd_locations.append(original_sgd_step)

            # sgd steps
            for sgd_learning_rate, sgd_steps in zip(sgd_learning_rates, sgd_step_locations):
                sgd_step = - sgd_learning_rate * direction_batch_directional_derivative
                sgd_steps.append(sgd_step)

            # pal steps:

            for pal_measuring_step_size, fb_pal_steps, in zip(pal_measuring_step_sizes, fb_pal_step_locations):
                lmu = mean_line_train_losses[get_nearest_index(sample_positions, pal_measuring_step_size)]
                l0 = mean_line_train_losses[get_nearest_index(sample_positions, 0)]
                b = mean_train_loss_directional_derivative
                a = (lmu - l0 - b * pal_measuring_step_size) / (
                        pal_measuring_step_size ** 2)
                if a > 0 and b < 0:
                    s_upd = -b / (2 * a)
                elif a <= 0 and b < 0:
                    s_upd = pal_measuring_step_size  # clone() since otherwise it's a reference to the measuring_step object
                else:
                    s_upd = 0
                fb_pal_steps.append(s_upd)

            pal_coefficients_list = []
            for pal_measuring_step_size, pal_steps, apal_steps, apal_adapt_coefficients, unsmoothed_apal_steps in zip(
                    pal_measuring_step_sizes,
                    pal_step_locations,
                    apal_step_locations,
                    apal_adapt_coefficients_list, apal_unsmoothed_step_locations):
                lmu = direction_batch_train_losses[get_nearest_index(sample_positions, pal_measuring_step_size)]
                l0 = direction_batch_train_losses[get_nearest_index(sample_positions, 0)]
                b = direction_batch_directional_derivative
                a = (lmu - l0 - direction_batch_directional_derivative * pal_measuring_step_size) / (
                        pal_measuring_step_size ** 2)
                if a > 0 and b < 0:
                    s_upd = -b / (2 * a)
                elif a <= 0 and b < 0:
                    s_upd = pal_measuring_step_size  # clone() since otherwise it's a reference to the measuring_step object
                else:
                    s_upd = 0
                pal_steps.append(s_upd)
                pal_coefficients_list.append([a, b, l0])
                # apal

                if (step % apal_resample_interval) == 0 or step == 1 or step == 100:
                    # lr = fb_pal_steps[-1] / -direction_batch_directional_derivative
                    lr = fb_pal_steps[-1]
                    if step >= 100:
                        unsmoothed_apal_steps.append(lr)
                        lr = np.nanmean(unsmoothed_apal_steps[-15:])
                    apal_adapt_coefficients[0] = lr

                else:
                    lr = apal_adapt_coefficients[0]
                apal_step = lr
                apal_steps.append(apal_step)

            ##############
            # distances
            ##############
            mean_min = mean_line_train_losses_min_data["min_position"]
            for sgd_steps, distance_sgd_step_to_mean_min in zip(sgd_step_locations, distances_sgd_step_to_mean_min):
                distance_sgd_step_to_mean_min.append(mean_min - sgd_steps[-1])
            for pal_steps, distance_pal_step_to_mean_min in zip(pal_step_locations, distances_pal_step_to_mean_min):
                distance_pal_step_to_mean_min.append(mean_min - pal_steps[-1])
            for fb_pal_steps, distance_fb_pal_step_to_mean_min in zip(fb_pal_step_locations,
                                                                      distances_fb_pal_step_to_mean_min):
                distance_fb_pal_step_to_mean_min.append(mean_min - fb_pal_steps[-1])
            for apal_steps, distance_apal_step_to_mean_min in zip(apal_step_locations, distances_apal_step_to_mean_min):
                distance_apal_step_to_mean_min.append(mean_min - apal_steps[-1])
            distances_direction_batch_min_to_mean_min.append(mean_min - direction_batch_losses_min_data["min_position"])
            distances_original_sgd_step_to_mean_min.append(mean_min - original_sgd_step)

            ##############
            # improvements
            ##############
            for sgd_steps, improvement_sgd in zip(sgd_step_locations, improvements_sgd):
                improvement_sgd.append(get_exact_improvement(sgd_steps[-1], sample_positions, mean_line_train_losses,
                                                             mean_line_train_losses_min_data))
            for pal_steps, improvement_pal in zip(pal_step_locations, improvements_pal):
                improvement_pal.append(get_exact_improvement(pal_steps[-1], sample_positions, mean_line_train_losses,
                                                             mean_line_train_losses_min_data))
            for fb_pal_steps, improvement_fb_pal in zip(fb_pal_step_locations, improvements_fb_pal):
                improvement_fb_pal.append(
                    get_exact_improvement(fb_pal_steps[-1], sample_positions, mean_line_train_losses,
                                          mean_line_train_losses_min_data))
            for apal_steps, improvement_apal in zip(apal_step_locations, improvements_apal):
                improvement_apal.append(get_exact_improvement(apal_steps[-1], sample_positions, mean_line_train_losses,
                                                              mean_line_train_losses_min_data))
            improvement_direction_batch_min.append(
                get_exact_improvement(direction_batch_min_locations[-1], sample_positions, mean_line_train_losses,
                                      mean_line_train_losses_min_data))
            improvement_real_loss_min.append(
                get_exact_improvement(mean_min, sample_positions, mean_line_train_losses,
                                      mean_line_train_losses_min_data))
            improvements_original_sgd.append(
                get_exact_improvement(original_sgd_locations[-1], sample_positions, mean_line_train_losses,
                                      mean_line_train_losses_min_data))

            if is_plot:
                plot_pure_line(sample_positions, min_line_train_losses, max_line_train_losses, q1_line_train_losses,
                               q2_line_train_losses, q3_line_train_losses, mean_line_train_losses,
                               direction_batch_train_losses,
                               plot_save_path, config_dict, step, [-0.5, 0.5], height_and_width, plot_pgf=True)
                plot_line_with_basic_approximations(sgd_step_locations, original_sgd_locations[-1],
                                                    pal_coefficients_list, sample_positions,
                                                    mean_line_train_losses,
                                                    direction_batch_train_losses,
                                                    mean_line_train_losses_min_data, direction_batch_losses_min_data,
                                                    direction_norm, directional_derivatives_direction_batch[-1]
                                                    , plot_save_path, config_dict, step, [-0.1, 0.5], height_and_width)

    save_dict = {
        "mean_min_locations": mean_min_locations,
        "original_sgd_locations": original_sgd_locations,
        "sgd_step_locations": sgd_step_locations,
        "pal_step_locations": pal_step_locations,
        "fb_pal_step_locations": fb_pal_step_locations,
        "apal_step_locations": apal_step_locations,

        "direction_batch_min_locations": direction_batch_min_locations,
        "distances_sgd_step_to_mean_min": distances_sgd_step_to_mean_min,
        "distances_pal_step_to_mean_min": distances_pal_step_to_mean_min,
        "distances_fb_pal_step_to_mean_min": distances_fb_pal_step_to_mean_min,
        "distances_apal_step_to_mean_min": distances_apal_step_to_mean_min,
        "distances_direction_batch_min_to_mean_min": distances_direction_batch_min_to_mean_min,
        "distances_original_sgd_step_to_mean_min": distances_original_sgd_step_to_mean_min,

        "improvements_sgd": improvements_sgd,
        "improvements_pal": improvements_pal,
        "improvements_fb_pal": improvements_fb_pal,
        "improvements_apal": improvements_apal,
        "improvement_direction_batch_min": improvement_direction_batch_min,
        "improvement_real_loss_min": improvement_real_loss_min,
        "improvements_original_sgd": improvements_original_sgd,

        "directional_derivatives_direction_batch": directional_derivatives_direction_batch,
        "directional_curvature_directional_batch": directional_curvature_directional_batch,
        "directional_derivatives_mean_train_loss": directional_derivatives_mean_train_loss,
        "directional_curvature_mean_train_loss": directional_curvature_mean_train_loss,

        "sample_variances_at_0": sample_variances_at_0,
        "variances_at_0": variances_at_0,
    }
    if not is_plot:
        write_line_statistics_data(save_dict, config_dict, data_save_path)
    return save_dict, config_dict


color_list = ["orange", "red", "blue", "violet", "green", "black", "purple"]
z_list = [2, 6, 1, 0, 5, 3, 4]
line_width = 2
smoothing = 25
scipping = 10
opacity = 0.75
batch_color_mapper = lambda x: plt.get_cmap("inferno")(
    np.log2(x) / np.log2(4000))  # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html


def plot_steps_per_steps(data_dict, save_path, config_dict, height_and_width, keys=None):
    print("_" * 50)
    print("Steps:")
    print("_" * 50)
    plt.figure()
    if keys == None:
        keys = ["mean_min_locations", "original_sgd_locations", "sgd_step_locations", "pal_step_locations",
                "fb_pal_step_locations", "apal_step_locations",
                "direction_batch_min_locations"]
    for key in keys:
        label_dict = {
            "sgd_step_locations": (0, "SGD $\lambda=$"),
            "mean_min_locations": (1, "full-batch loss (optimal)"),
            "pal_step_locations": (2, "PAL $\mu=$"),
            "apal_step_locations": (3, "FBPAL\&SGD $\mu=$"),
            "fb_pal_step_locations": (6, "FBPAL $\mu=$"),
            "direction_batch_min_locations": (4, "loss of direction defining batch"),
            "original_sgd_locations": (5, "SGD (original) $\lambda={}$".format(config_dict["ori_sgd_lr"])),
        }
        list_dict = {
            "sgd_step_locations": "sgd_lrs",
            "pal_step_locations": "pal_mus",
            "apal_step_locations": "pal_mus",
            "fb_pal_step_locations": "pal_mus",
        }
        data = data_dict[key]
        if not isinstance(data[0], list):
            data = np.array(data).ravel()
            data = moving_average(data, smoothing)
            x = range(0, len(data), scipping)
            data = data[x]
            sub_key, label = label_dict[key]
            plt.plot(x, data, label=label, zorder=z_list[sub_key], color=color_list[sub_key], linewidth=line_width,
                     alpha=opacity)
        else:
            sub_params = config_dict[list_dict[key]]
            for data_element, sub_param in zip(data, sub_params):
                data_element = np.array(data_element).ravel()
                data_element = moving_average(data_element, smoothing)
                sub_key, label = label_dict[key]
                label = "{0} {1}".format(label, np.round(sub_param, 8))
                x = range(0, len(data_element), scipping)
                data_element = data_element[x]
                plt.plot(x, data_element, label=label, zorder=z_list[sub_key], color=color_list[sub_key],
                         linewidth=line_width, alpha=opacity)
    plt.title("update steps (smoothed)")
    plt.xlabel("line number")
    plt.ylabel("update step length")
    plt.legend()
    additional_styleing = [height_and_width, "reverse legend",
                           "y tick label style={/pgf/number format/.cd,scaled y ticks = false,set thousands separator={},fixed,precision=4}"]
    tikzplotlib.save("{0}/{1}.pgf".format(save_path, "step_step"), strict=True,
                     extra_axis_parameters=additional_styleing)
    plt.savefig("{0}/{1}.png".format(save_path, "step_step"), dpi=200, bbox_inches='tight')
    plt.close()

    print("_" * 50)


def plot_improvement_over_batch_sizes(data_dict_list, config_dict_list, save_path, height_and_width, accumulate=False,
                                      keys=None):
    print("_" * 50)
    print("Plot Improvements over batch size:")
    print("_" * 50)

    label_dict = {
        "improvements_sgd": (0, "SGD $\lambda=$"),
        "improvement_real_loss_min": (1, "full-batch loss (optimal)"),
        "improvements_pal": (2, "PAL $\mu=$"),
        "improvements_apal": (3, "FBPAL&SGD $\mu=$"),
        "improvement_direction_batch_min": (4, "loss of direction defining batch"),
        "improvements_original_sgd": (
            5, "SGD (original) $\lambda={}$".format(np.round(config_dict_list[0]["ori_sgd_lr"], 4))),
    }
    list_dict = {
        "improvements_sgd": "sgd_lrs",
        "improvements_pal": "pal_mus",
        "improvements_apal": "pal_mus",
    }

    # improvements_apal
    if keys == None:
        keys = ["improvements_original_sgd", "improvements_sgd", "improvements_pal", "improvements_apal",
                "improvement_direction_batch_min"]
    for key in keys:
        if key in list_dict:
            sub_params = config_dict_list[0][list_dict[key]]
        else:
            sub_params = [""]
        for sub_param_num, sub_param in enumerate(sub_params):
            if sub_param is not "":
                sub_param = str(np.round(sub_param, 4))
            plt.figure()

            data = data_dict_list[0]["improvement_real_loss_min"]
            if accumulate:
                data = np.add.accumulate(data)
            else:
                data = moving_average(data, smoothing)
            x = range(0, len(data), scipping)
            data = data[x]
            sub_key, label = label_dict[key]
            color_num, label = label_dict["improvement_real_loss_min"]
            label = "full-batch loss (bs 4000)"
            plt.plot(x, data, label=label, zorder=z_list[1], color=color_list[color_num], linewidth=line_width)

            for data_dict, config_dict in zip(data_dict_list, config_dict_list):

                batch_size = str(config_dict["bs"])

                if sub_param == "":
                    data = data_dict[key]
                else:
                    data = data_dict[key][sub_param_num]
                if accumulate:
                    data = np.add.accumulate(data)
                else:
                    data = moving_average(data, smoothing)
                sub_key, label = label_dict[key]

                label = "batch size: " + str(batch_size)
                x = range(0, len(data), scipping)
                data = data[x]
                plt.plot(x, data, label=label, zorder=z_list[sub_key], color=batch_color_mapper(int(batch_size)),
                         linewidth=line_width)

            _, label = label_dict[key]
            label = "{0} {1}".format(label, sub_param)

            if accumulate:
                title = "accumulated loss improvement \\\\" + label_dict[key][1] + " " + sub_param
            else:
                title = "loss improvement \\\\" + label_dict[key][1] + " " + sub_param + " smoothed"

            plt.title(title)
            plt.xlabel("line number")
            plt.ylabel("full-batch loss improvement")
            if key == "improvements_original_sgd":
                plt.legend()

            label = clean_string(label)
            if accumulate:
                additional_styleing = ["align =center", height_and_width,
                                       "xtick = {-2000, 0, 2000, 4000, 6000, 8000, 10000}",
                                       "title style={font=\Large},"]
                tikzplotlib.save("{0}/{1}.pgf".format(save_path, label + "_improvements_step_accumulated"),
                                 extra_axis_parameters=additional_styleing)
                plt.savefig("{0}/{1}.png".format(save_path, label + "_improvements_step_accumulated"), dpi=200,
                            bbox_inches='tight')
            else:
                additional_styleing = [
                    "align=center",
                    "ytick={1e-4,1e-3,1e-2,1e-1,-1e-4,-1e-3,-1e-2,-1e-1,0}",
                    "yticklabels={$10^{-4}$,$10^{-3}$,$10^{-2}$,$10^{-1}$,$-10^{-4}$,$-10^{-3}$,$-10^{-2}$,$-10^{-1}$,$0$}",
                    "yticklabel style={/pgf/number format/.cd,int detect}",
                    "yminorgrids = true",
                    "ymajorgrids = true",
                    "xmajorgrids = true,", height_and_width, "xtick = {-2000, 0, 2000, 4000, 6000, 8000, 10000}",
                    "title style={font=\Large},"]
                tikzplotlib.save("{0}/{1}.pgf".format(save_path, label + "_improvements_step"),
                                 extra_axis_parameters=additional_styleing, strict=True)
                plt.yscale("symlog", linthresh=10E-5)
                plt.savefig("{0}/{1}.png".format(save_path, label + "_improvements_step"), dpi=200, bbox_inches='tight')
            plt.close()

            print("_" * 50)


def clean_string(s):
    translation_table1 = dict.fromkeys(map(ord, '!@#$\\ '), None)
    translation_table2 = dict.fromkeys(map(ord, '.'), ",")
    s = s.translate(translation_table1)
    s = s.translate(translation_table2)
    return s


def plot_improvements_per_steps(data_dict, save_path, config_dict, height_and_width, accumulate=False, keys=None, ):
    if accumulate:
        title = "accumulated loss improvement"
    else:
        title = "loss improvement (smoothed)"
    print("_" * 50)
    print("Improvements:")
    print("_" * 50)
    plt.figure()
    # improvements_apal
    if keys == None:
        keys = ["improvements_original_sgd", "improvements_sgd", "improvements_pal", "improvements_fb_pal",
                "improvements_apal", "improvement_direction_batch_min",
                "improvement_real_loss_min"]
    for key in keys:
        label_dict = {
            "improvements_sgd": (0, "SGD $\lambda=$"),
            "improvement_real_loss_min": (1, "full-batch loss (optimal)"),
            "improvements_pal": (2, "PAL $\mu=$"),
            "improvements_apal": (3, "FBPAL\&SGD $\mu=$"),
            "improvement_direction_batch_min": (4, "loss of direction defining batch"),
            "improvements_original_sgd": (5, "SGD (original) $\lambda={}$".format(config_dict["ori_sgd_lr"])),
            "improvements_fb_pal": (6, "FBPAL $\mu=$"),
        }

        list_dict = {
            "improvements_sgd": "sgd_lrs",
            "improvements_pal": "pal_mus",
            "improvements_apal": "pal_mus",
            "improvements_fb_pal": "pal_mus",
        }
        data = data_dict[key]
        if not isinstance(data[0], list):

            if accumulate:
                data = np.add.accumulate(data)
                alpha = 1.0
            else:
                data = moving_average(data, smoothing)
                alpha = opacity
            x = range(0, len(data), scipping)
            data = data[x]
            sub_key, label = label_dict[key]
            plt.plot(x, data, label=label, zorder=z_list[sub_key], color=color_list[sub_key], linewidth=line_width,
                     alpha=alpha)
        else:
            sub_params = config_dict[list_dict[key]]
            for data_element, sub_param in zip(data, sub_params):
                if accumulate:
                    data_element = np.add.accumulate(data_element)
                    alpha = 1.0
                else:
                    data_element = moving_average(data_element, smoothing)
                    alpha = opacity
                x = range(0, len(data_element), scipping)
                data_element = data_element[x]
                sub_key, label = label_dict[key]
                label = "{0} {1}".format(label, np.round(sub_param, 8))
                plt.plot(x, data_element, label=label, zorder=z_list[sub_key], color=color_list[sub_key],
                         linewidth=line_width, alpha=alpha)
    plt.title(title)
    plt.xlabel("line number")
    if accumulate:
        plt.ylabel("full-batch loss improvement")
    else:
        plt.ylabel("full-batch loss improvement (symlog)")
    if accumulate:
        additional_styleing = [height_and_width]
        tikzplotlib.save("{0}/{1}.pgf".format(save_path, "improvements_step_accumulated"),
                         extra_axis_parameters=additional_styleing)
        plt.savefig("{0}/{1}.png".format(save_path, "improvements_step_accumulated"), dpi=200, bbox_inches='tight')
    else:
        additional_styleing = [height_and_width,
                               "ytick={1e-4,1e-3,1e-2,1e-1,-1e-4,-1e-3,-1e-2,-1e-1,0},yticklabels={$10^{-4}$,$10^{-3}$,$10^{-2}$,$10^{-1}$,$-10^{-4}$,$-10^{-3}$,$-10^{-2}$,$-10^{-1}$,$0$},yticklabel style={/pgf/number format/.cd,int detect},yminorgrids = true,ymajorgrids = true,xmajorgrids = true,"]

        tikzplotlib.save("{0}/{1}.pgf".format(save_path, "improvements_step"),
                         extra_axis_parameters=additional_styleing, strict=True)
        plt.yscale("symlog", linthresh=10E-5)
        plt.savefig("{0}/{1}.png".format(save_path, "improvements_step"), dpi=200, bbox_inches='tight')
    plt.close()

    print("_" * 50)


def plot_distance_over_batch_sizes(data_dict_list, config_dict_list, save_path, height_and_width, accumulate=False,
                                   keys=None):
    print("_" * 50)
    print("Plot disntances over batch size:")
    print("_" * 50)
    label_dict = {
        "distances_sgd_step_to_mean_min": (0, "SGD $\lambda=$"),
        "distances_pal_step_to_mean_min": (2, "PAL $\mu=$"),
        "distances_apal_step_to_mean_min": (3, "FBPAL&SGD $\mu=$"),
        "distances_direction_batch_min_to_mean_min": (4, "loss of direction defining batch"),
        "distances_original_sgd_step_to_mean_min": (
            5, "SGD (original) $\lambda={}$".format(np.round(config_dict_list[0]["ori_sgd_lr"], 4))),
    }

    list_dict = {
        "distances_sgd_step_to_mean_min": "sgd_lrs",
        "distances_pal_step_to_mean_min": "pal_mus",
        "distances_apal_step_to_mean_min": "pal_mus",
    }

    if keys == None:
        keys = ["distances_original_sgd_step_to_mean_min", "distances_sgd_step_to_mean_min",
                "distances_pal_step_to_mean_min", "distances_apal_step_to_mean_min",
                "distances_direction_batch_min_to_mean_min"]
    for key in keys:
        if key in list_dict:
            sub_params = config_dict_list[0][list_dict[key]]
        else:
            sub_params = [""]
        for sub_param_num, sub_param in enumerate(sub_params):
            if sub_param is not "":
                sub_param = str(np.round(sub_param, 8))
            plt.figure()
            for data_dict, config_dict in zip(data_dict_list, config_dict_list):

                batch_size = str(config_dict["bs"])
                if sub_param == "":
                    data = data_dict[key]
                else:
                    data = data_dict[key][sub_param_num]
                if accumulate:
                    data = np.add.accumulate(data)
                else:
                    data = moving_average(data, smoothing)
                label = "batch size: {}".format(batch_size)
                x = range(0, len(data), scipping)
                data = data[x]
                plt.plot(x, data, label=label, zorder=z_list[1], color=batch_color_mapper(int(batch_size)),
                         linewidth=line_width)

            _, label = label_dict[key]
            label = "{0} {1}".format(label, sub_param)

            if accumulate:
                title = "accumulated distances to full-batch minimum \\\\ " + label_dict[key][1] + " " + sub_param
            else:
                title = "distances " + label_dict[key][1] + " " + sub_param + "\\\\ to full-batch minimum (smoothed)"
            plt.title(title)
            plt.xlabel("line number")
            plt.ylabel("distance to full-batch minimum")
            if key == "distances_direction_batch_min_to_mean_min":
                plt.legend()

            label = clean_string(label)
            if accumulate:

                additional_styleing = ["align =center", height_and_width,
                                       "xtick = {-2000, 0, 2000, 4000, 6000, 8000, 10000}",
                                       "title style={font=\Large},"]
                tikzplotlib.save("{0}/{1}.pgf".format(save_path, label + "_distances_step_accumulated"),
                                 extra_axis_parameters=additional_styleing)
                plt.savefig("{0}/{1}.png".format(save_path, label + "_distances_step_accumulated"), dpi=200,
                            bbox_inches='tight')
            else:
                additional_styleing = [
                    "align=center",
                    "ytick={1e-4,1e-3,1e-2,1e-1,-1e-4,-1e-3,-1e-2,-1e-1,0}",
                    "yticklabels={$10^{-4}$,$10^{-3}$,$10^{-2}$,$10^{-1}$,$-10^{-4}$,$-10^{-3}$,$-10^{-2}$,$-10^{-1}$,$0$}",
                    "yticklabel style={/pgf/number format/.cd,int detect}",
                    "yminorgrids = true",
                    "ymajorgrids = true",
                    "xmajorgrids = true,", height_and_width, "xtick = {-2000, 0, 2000, 4000, 6000, 8000, 10000}",
                    "title style={font=\Large},"]

                tikzplotlib.save("{0}/{1}.pgf".format(save_path, label + "_distances_step"),
                                 extra_axis_parameters=additional_styleing, strict=True)
                plt.yscale("symlog", linthresh=10E-5)
                plt.savefig("{0}/{1}.png".format(save_path, label + "_distances_step"), dpi=200, bbox_inches='tight')
            plt.close()

            print("_" * 50)


def plot_distances_per_steps(data_dict, save_path, config_dict, height_and_width, accumulate=False, keys=None):
    if accumulate:
        title = "accumulated distances to full-batch minimum"
    else:
        title = "distances to full-batch minimum (smoothed)"
    print("_" * 50)
    print("Distances:")
    print("_" * 50)
    plt.figure()
    if keys == None:
        keys = ["distances_original_sgd_step_to_mean_min", "distances_sgd_step_to_mean_min",
                "distances_pal_step_to_mean_min", "distances_apal_step_to_mean_min",
                "distances_fb_pal_step_to_mean_min", "distances_direction_batch_min_to_mean_min"]
    for key in keys:
        label_dict = {
            "distances_sgd_step_to_mean_min": (0, "SGD $\lambda=$"),
            "distances_pal_step_to_mean_min": (2, "PAL $\mu=$"),
            "distances_apal_step_to_mean_min": (3, "FBPAL\&SGD $\mu=$"),
            "distances_fb_pal_step_to_mean_min": (6, "FBPAL $\mu=$"),
            "distances_direction_batch_min_to_mean_min": (4, "loss of direction defining batch"),
            "distances_original_sgd_step_to_mean_min": (
                5, "SGD (original) $\lambda={}$".format(config_dict["ori_sgd_lr"])),
        }
        list_dict = {
            "distances_sgd_step_to_mean_min": "sgd_lrs",
            "distances_pal_step_to_mean_min": "pal_mus",
            "distances_apal_step_to_mean_min": "pal_mus",
            "distances_fb_pal_step_to_mean_min": "pal_mus",
        }
        data = data_dict[key]
        if not isinstance(data[0], list):

            if accumulate:
                data = np.add.accumulate(np.abs(data))
                alpha = 1.0
            else:
                data = moving_average(data, smoothing)
                alpha = opacity
            x = range(0, len(data), scipping)
            data = data[x]
            sub_key, label = label_dict[key]
            plt.plot(x, data, label=label, zorder=z_list[sub_key], color=color_list[sub_key], linewidth=line_width,
                     alpha=alpha)
        else:
            sub_params = config_dict[list_dict[key]]
            for data_element, sub_param in zip(data, sub_params):
                if accumulate:
                    data_element = np.add.accumulate(np.abs(data_element))
                    alpha = 1.0
                else:
                    data_element = moving_average(data_element, smoothing)
                    alpha = opacity
                sub_key, label = label_dict[key]
                label = "{0} {1}".format(label, np.round(sub_param, 8))
                x = range(0, len(data_element), scipping)
                data_element = data_element[x]
                plt.plot(x, data_element, label=label, zorder=z_list[sub_key], color=color_list[sub_key],
                         linewidth=line_width, alpha=alpha)
    plt.title(title)
    plt.xlabel("line number")
    if accumulate:
        plt.ylabel("distances to full-batch minimum")
    else:
        plt.ylabel("distances to full-batch minimum (symlog)")
    if accumulate:
        additional_styleing = [height_and_width]
        tikzplotlib.save("{0}/{1}.pgf".format(save_path, "distances_step_accumulated"),
                         extra_axis_parameters=additional_styleing)
        plt.savefig("{0}/{1}.png".format(save_path, "distances_step_accumulated"), dpi=200, bbox_inches='tight')
    else:
        additional_styleing = [height_and_width,
                               "ytick={1e-4,1e-3,1e-2,1e-1,-1e-4,-1e-3,-1e-2,-1e-1,0},yticklabels={$10^{-4}$,$10^{-3}$,$10^{-2}$,$10^{-1}$,$-10^{-4}$,$-10^{-3}$,$-10^{-2}$,$-10^{-1}$,$0$},yticklabel style={/pgf/number format/.cd,int detect},yminorgrids = true,ymajorgrids = true,xmajorgrids = true,"]

        tikzplotlib.save("{0}/{1}.pgf".format(save_path, "distances_step"),
                         extra_axis_parameters=additional_styleing, strict=True)
        plt.yscale("symlog", linthresh=10E-5)
        plt.savefig("{0}/{1}.png".format(save_path, "distances_step"), dpi=200, bbox_inches='tight')
    plt.close()

    print("_" * 50)


def plot_fb_step_to_min_to_db_grad_ratio(data_dict, save_path, config_dict, height_and_width):
    title = "Ratio of step to full-batch minimum and norm of dir. grad."
    print("_" * 50)
    print("directional derivatives:")
    print("_" * 50)
    plt.figure()
    base_data = np.array(data_dict["directional_derivatives_direction_batch"])

    data3 = np.array(data_dict["mean_min_locations"])
    y = data3 / (-base_data)
    y = moving_average(y, smoothing)
    x = range(0, len(y), scipping)
    y = y[x]
    plt.plot(x, y,
             linewidth=line_width, color="red")

    plt.title(title)
    plt.xlabel("line number")
    plt.ylabel("Ratio full-batch min. norm of dir. grad.")
    plt.legend()

    plt.ylim([-0.1, 0.3])
    additional_styleing = ["align =center"]
    tikzplotlib.save("{0}/{1}.pgf".format(save_path, "fb_min_step_ratio"),
                     extra_axis_parameters=additional_styleing, strict=True)

    additional_styleing = [height_and_width,
                           "y tick label style={/pgf/number format/.cd,scaled y ticks = false,set thousands separator={},fixed,precision=4}"]
    tikzplotlib.save("{0}/{1}.pgf".format(save_path, "fb_min_step_ratio"), strict=True,
                     extra_axis_parameters=additional_styleing)
    plt.savefig("{0}/{1}.png".format(save_path, "fb_min_step_ratio"), dpi=200, bbox_inches='tight')
    plt.close()
    print("_" * 50)


def plot_sgd_pal_data_dict(data_dict, save_path, config_dict, height_and_width):
    # matplotlib.use('Agg')
    save_path = os.path.join(save_path, dict_to_string(config_dict))
    os.makedirs(save_path, exist_ok=True)
    # plot improvements:
    plot_improvements_per_steps(data_dict, save_path, config_dict, height_and_width)
    plot_improvements_per_steps(data_dict, save_path, config_dict, height_and_width, accumulate=True)
    plot_distances_per_steps(data_dict, save_path, config_dict, height_and_width)
    plot_distances_per_steps(data_dict, save_path, config_dict, height_and_width, accumulate=True)
    plot_steps_per_steps(data_dict, save_path, config_dict, height_and_width)
    plot_fb_step_to_min_to_db_grad_ratio(data_dict, save_path, config_dict, height_and_width)


def plot_directional_curvatures(data_dict_list, config_dict_list, save_path):
    title = "directional curvature depending \\\\ on the batch size"
    print("_" * 50)
    print("directional curvature:")
    print("_" * 50)
    plt.figure()

    data = data_dict_list[0]["directional_curvature_mean_train_loss"]
    data = moving_average(data, smoothing)
    x = range(0, len(data), scipping)
    data = data[x]
    plt.plot(x, data, label="full-batch size (4000)",
             linewidth=line_width)

    for data_dict, config_dict in zip(data_dict_list, config_dict_list):
        batch_size = config_dict["bs"]
        directional_derivatives_batch = data_dict["directional_curvature_directional_batch"]
        directional_derivatives_batch = moving_average(directional_derivatives_batch, smoothing)
        directional_derivatives_batch = directional_derivatives_batch[x]
        plt.plot(x, directional_derivatives_batch, label=str(batch_size), linewidth=line_width,
                 color=batch_color_mapper(batch_size))

    plt.title(title)
    plt.xlabel("line number")
    plt.ylabel("directional curvature")
    plt.legend()

    additional_styleing = ["align =center"]
    tikzplotlib.save("{0}/{1}.pgf".format(save_path, "directional_curvatures"),
                     extra_axis_parameters=additional_styleing)
    plt.savefig("{0}/{1}.png".format(save_path, "directional_curvatures"), dpi=200, bbox_inches='tight')
    plt.close()
    print("_" * 50)


def plot_directional_curvature_ratios(data_dict_list, config_dict_list, save_path):
    title = "ratio of \\\\ directional curvature  and batch size"
    print("_" * 50)
    print("directional curvature ratio:")
    print("_" * 50)
    plt.figure()

    for data_dict, config_dict in zip(data_dict_list, config_dict_list):
        if config_dict["bs"] == config_dict["bs_ori"]:
            base_data = data_dict["directional_curvature_directional_batch"]
            base_data = moving_average(base_data, smoothing)

    data = data_dict_list[0]["directional_curvature_mean_train_loss"]
    data = moving_average(data, smoothing)
    x = range(0, len(data), scipping)
    data = data[x]
    base_data = base_data[x]
    plt.plot(x, data / base_data,
             label="full-batch size (4000) scale: " + str(4000 / config_dict["bs_ori"]),
             linewidth=line_width)

    for data_dict, config_dict in zip(data_dict_list, config_dict_list):
        batch_size = config_dict["bs"]
        key = "directional_curvature_directional_batch"
        directional_derivatives_batch = data_dict[key]
        directional_derivatives_batch = moving_average(directional_derivatives_batch, smoothing)
        directional_derivatives_batch = directional_derivatives_batch[x]
        plt.plot(x, directional_derivatives_batch / base_data,
                 label="bs:" + str(batch_size) + " scale: " + str(config_dict["bs"] / config_dict["bs_ori"]),
                 linewidth=line_width, color=batch_color_mapper(batch_size))

    plt.title(title)
    plt.xlabel("line number")
    plt.ylabel("ratio dir. curvature & batch size")
    plt.legend()

    additional_styleing = ["align =center"]
    tikzplotlib.save("{0}/{1}.pgf".format(save_path, "directional_curvature_ratio"),
                     extra_axis_parameters=additional_styleing)
    plt.savefig("{0}/{1}.png".format(save_path, "directional_curvature_ratio"), dpi=200, bbox_inches='tight')
    plt.close()
    print("_" * 50)


def plot_directional_derivatives(data_dict_list, config_dict_list, save_path, height_and_width):
    title = "abs. of directional derivative  ($=||g||$) \\\\  depending on the batch size"
    print("_" * 50)
    print("directional derivatives:")
    print("_" * 50)
    plt.figure()

    data = data_dict_list[0]["directional_derivatives_mean_train_loss"]
    data = moving_average(data, smoothing)
    x = range(0, len(data), scipping)
    data = np.abs(data[x])
    plt.plot(x, data, label="full-batch size (4000)",
             linewidth=line_width, color="red")

    for data_dict, config_dict in zip(data_dict_list, config_dict_list):
        batch_size = config_dict["bs"]
        directional_derivatives_batch = data_dict["directional_derivatives_direction_batch"]
        directional_derivatives_batch = moving_average(directional_derivatives_batch, smoothing)
        directional_derivatives_batch = directional_derivatives_batch[x]
        norm_directional_derivatives_batch = np.abs(directional_derivatives_batch)

        plt.plot(x, norm_directional_derivatives_batch, label=str(batch_size), linewidth=line_width,
                 color=batch_color_mapper(batch_size))

    plt.title(title)
    plt.xlabel("line number")
    plt.ylabel("abs. of directional derivative")

    additional_styleing = ["align =center", height_and_width, "xtick = {-2000, 0, 2000, 4000, 6000, 8000, 10000}",
                           "title style={font=\Large},"]
    tikzplotlib.save("{0}/{1}.pgf".format(save_path, "directional_derivatives"),
                     extra_axis_parameters=additional_styleing)
    plt.savefig("{0}/{1}.png".format(save_path, "directional_derivatives"), dpi=200, bbox_inches='tight')
    plt.close()
    print("_" * 50)


def plot_directional_derivatives_ratios(data_dict_list, config_dict_list, save_path, height_and_width):
    title = "ratio of directional derivatives at several \\\\  batch sizes & directional derivatives at bs 128"
    print("_" * 50)
    print("directional derivatives:")
    print("_" * 50)
    plt.figure()

    for data_dict, config_dict in zip(data_dict_list, config_dict_list):
        if config_dict["bs"] == config_dict["bs_ori"]:
            base_data = data_dict["directional_derivatives_direction_batch"]
            base_data = moving_average(base_data, smoothing)

    data = data_dict_list[0]["directional_derivatives_mean_train_loss"]
    data = moving_average(data, smoothing)
    x = range(0, len(data), scipping)
    data = data[x]
    base_data = base_data[x]
    plt.plot(x, data / base_data,
             label="full-bs e.r.: " + str(config_dict["bs_ori"] / 4000),
             linewidth=line_width, color="red")

    for data_dict, config_dict in zip(data_dict_list, config_dict_list):
        batch_size = config_dict["bs"]
        directional_derivatives_batch = data_dict["directional_derivatives_direction_batch"]
        directional_derivatives_batch = moving_average(directional_derivatives_batch, smoothing)
        directional_derivatives_batch = directional_derivatives_batch[x]
        plt.plot(x, directional_derivatives_batch / base_data,
                 label="bs: " + str(batch_size) + " e.r.: " + str(config_dict["bs_ori"] / config_dict["bs"]),
                 linewidth=line_width, color=batch_color_mapper(batch_size))

    plt.title(title)
    plt.xlabel("line number")
    plt.ylabel("ratio of dir. deriv. at several bs")
    plt.yscale("log", base=2)
    plt.legend(loc="upper right")

    additional_styleing = ["align =center", height_and_width, "xtick = {-2000, 0, 2000, 4000, 6000, 8000, 10000}",
                           "title style={font=\Large},"]
    tikzplotlib.save("{0}/{1}.pgf".format(save_path, "directional_derivatives_ratio"),
                     extra_axis_parameters=additional_styleing)
    plt.savefig("{0}/{1}.png".format(save_path, "directional_derivatives_ratio"), dpi=200, bbox_inches='tight')
    plt.close()
    print("_" * 50)


def plot_fb_step_to_min_to_db_grad_ratio_batches(data_dict_list, config_dict_list, save_path):
    title = "Ratio step to full-batch minimum and norm of dir. grad. over batch sizes"
    print("_" * 50)
    print("directional derivatives:")
    print("_" * 50)
    plt.figure()

    base_data3 = np.array(data_dict_list[0]["mean_min_locations"])

    for data_dict, config_dict in zip(data_dict_list, config_dict_list):
        data = np.array(data_dict["directional_derivatives_direction_batch"])

        y = base_data3 / -data
        y = moving_average(y, smoothing)
        x = range(0, len(y), scipping)
        y = y[x]
        batch_size = config_dict["bs"]
        plt.plot(x, y,
                 label="bs:" + str(batch_size) + " scale: " + str(config_dict["bs"] / config_dict["bs_ori"]),
                 linewidth=line_width, color=batch_color_mapper(batch_size))

    plt.title(title)
    plt.xlabel("line number")
    plt.ylabel("Ratio full-batch min. norm of dir. grad.")
    plt.legend()

    additional_styleing = ["align =center,ymax=1.5"]
    tikzplotlib.save("{0}/{1}.pgf".format(save_path, "fb_min_step_ratio_batches"),
                     extra_axis_parameters=additional_styleing)
    plt.savefig("{0}/{1}.png".format(save_path, "fb_min_step_ratio_batches"), dpi=200, bbox_inches='tight')
    plt.close()
    print("_" * 50)


def plot_variances(data_dict_list, config_dict_list, save_path):
    title = "variance at line position 0"
    print("_" * 50)
    print("loss variance:")
    print("_" * 50)
    plt.figure()
    line_width = 1.5
    for data_dict, config_dict in zip(data_dict_list, config_dict_list):
        batch_size = config_dict["bs"]
        batch_variance_at_0 = data_dict["sample_variances_at_0"]
        batch_variance_at_0 = moving_average(batch_variance_at_0, smoothing)
        x = range(len(batch_variance_at_0))
        plt.plot(x, batch_variance_at_0, label="batch variance" + str(batch_size), linewidth=line_width, color="green")
    variances_at_0 = data_dict["variances_at_0"]
    variances_at_0 = moving_average(variances_at_0, smoothing)
    x = range(len(variances_at_0))
    plt.plot(x, variances_at_0, label="variance", linewidth=line_width, color="red")

    plt.title(title)
    plt.xlabel("line number")
    plt.ylabel("loss variance")
    plt.legend()
    plt.yscale("log")

    tikzplotlib.save("{0}/{1}.pgf".format(save_path, "loss_variance"))
    plt.savefig("{0}/{1}.png".format(save_path, "loss_variance"), dpi=200, bbox_inches='tight')

    print("_" * 50)
