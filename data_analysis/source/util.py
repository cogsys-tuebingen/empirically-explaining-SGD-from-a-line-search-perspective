import copy
import itertools
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


def moving_average(a, n):
    return np.convolve(a, np.ones(n), 'valid') / n


def create_save_dirs(data_path, group_sting=""):
    save_path = data_path + "_result"
    data_save_path = os.path.join(save_path, "data", group_sting)
    plot_save_path = os.path.join(save_path, "line_plots", group_sting)
    statistics_plot_save_path = os.path.join(save_path, "statistics_plots", group_sting)
    mean_line_save_path = os.path.join(save_path, "mean_line_data", group_sting)
    batch_compare_path = os.path.join(statistics_plot_save_path, "batch_size_comparison", group_sting)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(data_save_path, exist_ok=True)
    os.makedirs(plot_save_path, exist_ok=True)
    os.makedirs(statistics_plot_save_path, exist_ok=True)
    os.makedirs(mean_line_save_path, exist_ok=True)
    os.makedirs(batch_compare_path, exist_ok=True)

    return save_path, data_save_path, plot_save_path, statistics_plot_save_path, mean_line_save_path, batch_compare_path


def sort_key_group(text):
    return int(text.split(".")[0].split("_")[3][:])


def create_and_sort_position_groups(data_files):
    groups = []
    for data_file in data_files:
        position = data_file.split("_")[2]
        found = False
        for group in groups:
            if group[0].split("_")[2] == position:
                group.append(data_file)
                found = True
                break
        if not found:
            groups.append([data_file])
    for group in groups:
        group.sort(key=sort_key_group)
    return groups


def load_and_sort_data_files(data_load_path):
    data_files = os.listdir(data_load_path)
    data_files.sort(key=sort_key)
    return data_files


def get_mean_square_error(x, y, coefficients):
    assert len(x) == len(y)
    assert len(x) > 0
    return np.sum((np.polyval(coefficients, x) - y) ** 2) / len(x)


def get_root_mean_square_error(x, y, coefficients):
    return np.sqrt(get_mean_square_error(x, y, coefficients))


def get_mean_absolut_error(x, y, coefficients):
    assert len(x) == len(y)
    assert len(x) > 0
    return np.sum(np.abs((np.polyval(coefficients, x) - y))) / len(x)


def get_direcional_derivative(losses, sample_positions):
    zero_index = np.where(sample_positions == 0)[0][0]
    directional_derivatives = np.gradient(losses[zero_index - 2:zero_index + 3],
                                          sample_positions[zero_index - 2:zero_index + 3])
    directional_curvatures = np.gradient(directional_derivatives[:], sample_positions[zero_index - 2:zero_index + 3])
    return directional_derivatives[2], directional_curvatures[2]


def get_loss_variance_at(position, batched_line_train_losses, sample_positions):
    index = get_nearest_index(sample_positions, position)
    losses = batched_line_train_losses[:, index]
    variance = np.nanvar(losses)
    return variance


def load_line_data(data_load_path, dir_batch_data_load_path, line_data_file, dir_batch_file):
    with open(os.path.join(data_load_path, line_data_file), 'rb') as f:
        data = np.load(f, allow_pickle=True)
        if len(data) == 7:
            train_line_losses, val_line_losses, sample_positions, direction_batch_number, sgd_update_step, direction_norm, is_extrapolation = data
        elif len(data) == 8:
            train_line_losses, val_line_losses, sample_positions, direction_batch_number, sgd_update_step, direction_norm, is_extrapolation, direction_batch_number2 = data

    sample_positions = np.array(sample_positions)
    train_line_losses = np.array(train_line_losses)
    val_line_losses = np.array(val_line_losses)

    direction_norm = direction_norm.item()
    with open(os.path.join(dir_batch_data_load_path, dir_batch_file), 'rb') as f:
        direction_batch_losses = np.load(f, allow_pickle=True)[0]

    return train_line_losses, val_line_losses, sample_positions, direction_batch_losses, sgd_update_step, direction_norm, is_extrapolation


def load_line_data_no_dir_batch_data(data_load_path, line_data_file):
    with open(os.path.join(data_load_path, line_data_file), 'rb') as f:
        data = np.load(f, allow_pickle=True)
        train_line_losses, val_line_losses, sample_positions, direction_batch_losses, sgd_update_step, direction_norm, is_extrapolation, direction_batch_number2 = data
    sample_positions = np.array(sample_positions)
    train_line_losses = np.array(train_line_losses)
    val_line_losses = np.array(val_line_losses)

    direction_norm = direction_norm.item()

    return train_line_losses, val_line_losses, sample_positions, direction_batch_losses, sgd_update_step, direction_norm, is_extrapolation, direction_batch_number2


def get_value_index_location_and_error_of_min(losses, sample_positions):
    min_index = int(np.where(losses == np.amin(losses))[0][0])
    min_position = sample_positions[min_index]
    left_index = min_index - 1 if min_index - 1 > 0 else 0
    right_index = min_index + 1 if min_index + 1 < len(sample_positions) - 1 else len(sample_positions) - 1
    position_uncertainty_min = np.abs(sample_positions[left_index] - min_position)
    position_uncertainty_max = np.abs(sample_positions[right_index] - min_position)
    loss_value_at_min = losses[min_index]
    return {"loss_value": loss_value_at_min, "min_index": min_index, "min_position": min_position,
            "position_uncertainty_min": position_uncertainty_min, "position_uncertainty_max": position_uncertainty_max}


def get_relative_improvement(step, sample_positions, losses, losses_min_data):
    l0 = losses[get_nearest_index(sample_positions, 0)]
    lstep = losses[get_nearest_index(sample_positions, step)]
    lmin = losses_min_data["loss_value"]
    return (l0 - lstep) / (l0 - lmin + 10E-8)


def get_exact_improvement(step, sample_positions, losses, losses_min_data):
    l0 = losses[get_nearest_index(sample_positions, 0)]
    lstep = losses[get_nearest_index(sample_positions, step)]
    return l0 - lstep


def get_accumulated_data_array(data):
    return np.add.accumulate(data)


def plot_all_losses_on_line(sample_positions, config_dict, line_losses, min_line_losses, max_line_losses,
                            direction_batch_losses, mean_line_losses, save_path, step):
    save_path = os.path.join(save_path, dict_to_string(config_dict))
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(20, 20))
    plt.fill_between(sample_positions, min_line_losses, max_line_losses, color="blue", alpha=0.2,
                     linewidth=1)

    for i, row in enumerate(line_losses):

        if (i + 1) % 10 == 0:
            plt.plot(sample_positions, row, linewidth=0.5)
            plt.plot(sample_positions, mean_line_losses, color="red", linestyle="dashed", linewidth=1)
            plt.plot(sample_positions, direction_batch_losses, color="green", linestyle="dashed", linewidth=1)

            plt.show(block=True)
            plt.close()
            plt.figure(figsize=(20, 20))
        else:
            plt.plot(sample_positions, row, linewidth=0.5)
    plt.plot(sample_positions, mean_line_losses, color="red", linewidth=1)
    plt.plot(sample_positions, direction_batch_losses, color="green", linewidth=1)

    plt.savefig("{0}/train_line_{1:d}_all_losses.png".format(save_path, step), dpi=200,
                bbox_inches='tight')

    plt.close()


def plot_pure_line(sample_positions, min_line_losses, max_line_losses, first_quartile_loss, second_quartile_loss,
                   third_quartile_loss, mean_line_losses, direction_batch_losses, save_path, config_dict, step, limits,
                   height_and_width, plot_pgf=True):
    print(step)
    matplotlib.use("TkAgg")
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
    })

    limits_indexes = get_nearest_indexes(sample_positions, limits)
    direction_batch_losses = direction_batch_losses[limits_indexes[0]:limits_indexes[1]]
    min_line_losses = min_line_losses[limits_indexes[0]:limits_indexes[1]]
    max_line_losses = max_line_losses[limits_indexes[0]:limits_indexes[1]]
    mean_line_losses = mean_line_losses[limits_indexes[0]:limits_indexes[1]]
    first_quartile_loss = first_quartile_loss[limits_indexes[0]:limits_indexes[1]]
    second_quartile_loss = second_quartile_loss[limits_indexes[0]:limits_indexes[1]]
    third_quartile_loss = third_quartile_loss[limits_indexes[0]:limits_indexes[1]]
    sample_positions = sample_positions[limits_indexes[0]:limits_indexes[1]]

    linewidth = 3
    save_path = os.path.join(save_path, dict_to_string(config_dict))
    os.makedirs(save_path, exist_ok=True)
    plt.figure()
    plt.fill_between(sample_positions, min_line_losses, max_line_losses, color="blue", alpha=0.2,
                     linewidth=1, label="fill between min. and max. batch losses")

    plt.plot(sample_positions, first_quartile_loss, color="black", linewidth=linewidth / 2)
    plt.plot(sample_positions, second_quartile_loss, color="black", linewidth=linewidth / 2)
    plt.plot(sample_positions, third_quartile_loss, color="black", linewidth=linewidth / 2,
             label="quartiles of batch losses")
    plt.plot(sample_positions, direction_batch_losses, color="green", linewidth=linewidth,
             label="loss of direction defining batch")
    plt.plot(sample_positions, mean_line_losses, color="red", linewidth=linewidth, label="full batch loss")

    y_lim = min(min(direction_batch_losses), min(min_line_losses)) - min(min(direction_batch_losses),
                                                                         min(min_line_losses)) * 0.2

    ax = plt.gca()
    plt.xlabel("step on line (s)")
    plt.ylabel("loss")
    ymin, ymax = plt.gca().get_ylim()
    ax.set_ylim((y_lim, min(5, ymax)))
    plt.title("line number: " + str(step))

    if step < 50:
        plt.legend()

    plt.savefig("{0}/pure_line_{1:d}.png".format(save_path, step), dpi=200,
                bbox_inches='tight')
    if plot_pgf:
        config = "y tick label style={/pgf/number format/.cd,scaled y ticks = false,set thousands separator={},fixed,precision=4}"
        tikzplotlib.save("{0}/pure_line_{1:d}.pgf".format(save_path, step),
                         extra_axis_parameters=[height_and_width, config])  # textsize is not used by tkzplotlib

    plt.close()


def plot_line_with_basic_approximations(sgd_update_steps_list, ori_sgd_step, pal_coefficients_list, sample_positions,
                                        mean_line_losses, direction_batch_losses, mean_line_losses_min_data,
                                        direction_batch_losses_min_data,
                                        direction_norm, directional_derivative, save_path, config_dict, step, limits,
                                        height_and_width):
    print(step)
    matplotlib.use("TkAgg")
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
    })
    sgd_learning_rates = config_dict["sgd_lrs"]
    pal_measuring_step_sizes = config_dict["pal_mus"]
    ori_sgd_lr = config_dict["ori_sgd_lr"]

    limits_indexes = get_nearest_indexes(sample_positions, limits)
    direction_batch_losses = direction_batch_losses[limits_indexes[0]:limits_indexes[1]]
    mean_line_losses = mean_line_losses[limits_indexes[0]:limits_indexes[1]]
    sample_positions = sample_positions[limits_indexes[0]:limits_indexes[1]]

    linewidth = 3.0
    dashed_line_boundary = 5
    label_size = 100
    save_path = os.path.join(save_path, dict_to_string(config_dict))
    os.makedirs(save_path, exist_ok=True)
    fig = plt.figure()

    x = sample_positions
    resolution = (max(x) - min(x)) / 200
    sgd_approx_x = batch_approx_x = np.arange(min(x), max(x), resolution)

    # ori_sgd approx
    _0index = np.where(sample_positions == 0)[0]
    c = direction_batch_losses[_0index]
    b = -direction_norm
    a = -b / (ori_sgd_step * 2)
    sgd_parabola_coefficients = [a, b, c]
    sgd_approx_y = np.polyval(sgd_parabola_coefficients, sgd_approx_x)
    min_sgd_value = min(sgd_approx_y)
    plt.plot(sgd_approx_x, sgd_approx_y, color="black", linewidth=linewidth,
             label="parabolic approximation SGD $\lambda={}$ (original)".format(ori_sgd_lr))
    sgd_min_y_index = sgd_approx_y.argmin()
    plt.scatter([sgd_approx_x[sgd_min_y_index], 0], [sgd_approx_y[sgd_min_y_index], -5],
                color="black", marker="o", s=label_size, zorder=100)
    plt.plot([sgd_approx_x[sgd_min_y_index], sgd_approx_x[sgd_min_y_index]],
             [-1, dashed_line_boundary], color="black", linewidth=linewidth,
             linestyle='dashed')

    #  sgd approx

    c = direction_batch_losses[_0index]
    b = directional_derivative

    for sgd_update_steps, sgd_learning_rate in zip(sgd_update_steps_list, sgd_learning_rates):
        a = -b / (sgd_update_steps[-1] * 2)
        sgd_parabola_coefficients = [a, b, c]
        sgd_approx_y = np.polyval(sgd_parabola_coefficients, sgd_approx_x)
        min_sgd_value = min_sgd_value if min_sgd_value < min(sgd_approx_y) else min(sgd_approx_y)
        plt.plot(sgd_approx_x, sgd_approx_y, color="orange", linewidth=linewidth,
                 label="parabolic approximation SGD $\lambda={}$ ".format(sgd_learning_rate))
        sgd_min_y_index = sgd_approx_y.argmin()
        plt.scatter([sgd_approx_x[sgd_min_y_index], 0], [sgd_approx_y[sgd_min_y_index], -5],
                    color="orange", marker="o", s=label_size, zorder=100)
        plt.plot([sgd_approx_x[sgd_min_y_index], sgd_approx_x[sgd_min_y_index]],
                 [-1, dashed_line_boundary], color="orange", linewidth=linewidth,
                 linestyle='dashed')

    # pal approx
    min_pal_value = np.inf
    for pal_coefficients, pal_measuring_step_size in zip(pal_coefficients_list, pal_measuring_step_sizes):
        batch_approx_y = np.polyval(pal_coefficients, batch_approx_x)
        min_pal_value = min_pal_value if min_pal_value < min(batch_approx_y) else min(batch_approx_y)
        plt.plot(batch_approx_x, batch_approx_y, color="blue", linewidth=linewidth,
                 label="parabolic approximation PAL $\mu={}$ ".format(pal_measuring_step_size))
        min_y_index = batch_approx_y.argmin()
        plt.scatter([batch_approx_x[min_y_index], 0], [batch_approx_y[min_y_index], -5],
                    color="blue", marker="o", s=label_size, zorder=100)
        plt.plot([batch_approx_x[min_y_index], batch_approx_x[min_y_index]],
                 [-1, dashed_line_boundary], color="blue", linewidth=linewidth,
                 linestyle='dashed')

    # FOR SOME REASON DOES TIKZPLOTLIB DEMANDS MORE THAN ONE SCATTER POINT
    plt.scatter([mean_line_losses_min_data["min_position"], 0], [mean_line_losses_min_data["loss_value"], -5],
                color="red",
                marker="D", s=label_size)
    plt.scatter([direction_batch_losses_min_data["min_position"], 0],
                [direction_batch_losses_min_data["loss_value"], -5],
                color="green",
                marker="X", s=label_size)

    y_lim = (min(min_sgd_value, min_pal_value) - min(min_sgd_value, min_pal_value) * 0.1, max(mean_line_losses))
    plt.plot(sample_positions, mean_line_losses, color="red", linewidth=linewidth, label="full batch loss")
    # plt.plot(sample_positions, mean_line_val_losses, color="purple", linewidth=linewidth, label="mean val loss")
    plt.plot(sample_positions, direction_batch_losses, color="green", linewidth=linewidth,
             label="loss of direction defining batch")

    plt.plot([mean_line_losses_min_data["min_position"], mean_line_losses_min_data["min_position"]],
             [-1, dashed_line_boundary], color="red", linewidth=linewidth,
             linestyle='dashed')

    plt.plot([direction_batch_losses_min_data["min_position"], direction_batch_losses_min_data["min_position"]],
             [-1, dashed_line_boundary], color="green", linewidth=linewidth,
             linestyle='dashed')
    plt.plot([0], [0], color="gray", linewidth=linewidth,
             linestyle='dashed', label="minima locations")
    ax = plt.gca()
    plt.xlabel("step on line")
    plt.ylabel("loss")
    ax.set_ylim(y_lim)
    plt.title("line number " + str(step))
    if step < 11:
        plt.legend()
    config = "y tick label style={/pgf/number format/.cd,scaled y ticks = false,set thousands separator={},fixed,precision=4}"
    tikzplotlib.save("{0}/sgd_pal_line_{1:d}.pgf".format(save_path, step),
                     extra_axis_parameters=["legend style={font=\\normalsize}", config, height_and_width])
    plt.close()


def plot_sgd_pal_line(sgd_update_steps_list, ori_sgd_step, pal_coefficients_list, apal_adapt_coefficients,
                      sample_positions,
                      min_line_losses, max_line_losses, mean_line_losses, direction_batch_losses,
                      mean_line_losses_min_data, direction_batch_losses_min_data,
                      direction_norm, directional_derivative, save_path, config_dict, step, limits):
    limits_indexes = get_nearest_indexes(sample_positions, limits)
    direction_batch_losses = direction_batch_losses[limits_indexes[0]:limits_indexes[1]]
    min_line_losses = min_line_losses[limits_indexes[0]:limits_indexes[1]]
    max_line_losses = max_line_losses[limits_indexes[0]:limits_indexes[1]]
    mean_line_losses = mean_line_losses[limits_indexes[0]:limits_indexes[1]]
    sample_positions = sample_positions[limits_indexes[0]:limits_indexes[1]]

    linewidth = 1.5
    save_path = os.path.join(save_path, dict_to_string(config_dict))
    os.makedirs(save_path, exist_ok=True)
    plt.figure()
    plt.fill_between(sample_positions, min_line_losses, max_line_losses, color="blue", alpha=0.2,
                     linewidth=1)
    plt.plot(sample_positions, mean_line_losses, color="red", linewidth=linewidth, label="mean train loss")
    plt.plot(sample_positions, direction_batch_losses, color="green", linewidth=linewidth, label="batch loss")
    x = sample_positions
    resolution = (max(x) - min(x)) / 10000
    sgd_approx_x = batch_approx_x = np.arange(min(x), max(x), resolution)

    # ori_sgd approx
    _0index = np.where(sample_positions == 0)[0]
    c = direction_batch_losses[_0index]
    b = -direction_norm
    a = -b / (ori_sgd_step * 2)
    sgd_parabola_coefficients = [a, b, c]
    sgd_approx_y = np.polyval(sgd_parabola_coefficients, sgd_approx_x)
    min_sgd_value = min(sgd_approx_y)
    plt.plot(sgd_approx_x, sgd_approx_y, color="orange", linewidth=linewidth, label="original sgd parabola")
    sgd_min_y_index = sgd_approx_y.argmin()
    plt.scatter(sgd_approx_x[sgd_min_y_index], sgd_approx_y[sgd_min_y_index],
                color="orange", marker="o", s=50, zorder=100)
    plt.plot([sgd_approx_x[sgd_min_y_index], sgd_approx_x[sgd_min_y_index]],
             [-1, sgd_approx_y[sgd_min_y_index]], color="orange", linewidth=linewidth,
             linestyle='dashed')

    #  sgd approx

    c = direction_batch_losses[_0index]
    b = directional_derivative
    for sgd_update_steps in sgd_update_steps_list:
        a = -b / (sgd_update_steps[-1] * 2)
        sgd_parabola_coefficients = [a, b, c]
        sgd_approx_y = np.polyval(sgd_parabola_coefficients, sgd_approx_x)
        min_sgd_value = min_sgd_value if min_sgd_value < min(sgd_approx_y) else min(sgd_approx_y)
        plt.plot(sgd_approx_x, sgd_approx_y, color="black", linewidth=linewidth, label="sgd parabola")
        sgd_min_y_index = sgd_approx_y.argmin()
        plt.scatter(sgd_approx_x[sgd_min_y_index], sgd_approx_y[sgd_min_y_index],
                    color="black", marker="o", s=50, zorder=100)
        plt.plot([sgd_approx_x[sgd_min_y_index], sgd_approx_x[sgd_min_y_index]],
                 [-1, sgd_approx_y[sgd_min_y_index]], color="black", linewidth=linewidth,
                 linestyle='dashed')

    # pal approx
    min_pal_value = np.inf
    for pal_coefficients in pal_coefficients_list:
        batch_approx_y = np.polyval(pal_coefficients, batch_approx_x)
        min_pal_value = min_pal_value if min_pal_value < min(batch_approx_y) else min(batch_approx_y)
        plt.plot(batch_approx_x, batch_approx_y, color="blue", linewidth=linewidth, label="pal parabola")
        min_y_index = batch_approx_y.argmin()
        plt.scatter(batch_approx_x[min_y_index], batch_approx_y[min_y_index],
                    color="blue", marker="o", s=50, zorder=100)
        plt.plot([batch_approx_x[min_y_index], batch_approx_x[min_y_index]],
                 [-1, batch_approx_y[min_y_index]], color="blue", linewidth=linewidth,
                 linestyle='dashed')

    for pal_coefficients, adaptation_factors in zip(pal_coefficients_list, apal_adapt_coefficients):
        pal_coefficients[0] *= adaptation_factors[0]
        pal_coefficients[1] *= adaptation_factors[1]
        batch_approx_y = np.polyval(pal_coefficients, batch_approx_x)
        min_pal_value = min_pal_value if min_pal_value < min(batch_approx_y) else min(batch_approx_y)
        plt.plot(batch_approx_x, batch_approx_y, color="pink", linestyle="dashed", linewidth=linewidth,
                 label="apal parabola")
        min_y_index = batch_approx_y.argmin()
        plt.scatter(batch_approx_x[min_y_index], batch_approx_y[min_y_index],
                    color="pink", marker="o", s=50, zorder=100)
        plt.plot([batch_approx_x[min_y_index], batch_approx_x[min_y_index]],
                 [-1, batch_approx_y[min_y_index]], color="pink", linewidth=linewidth,
                 linestyle='dashed')

    plt.scatter([mean_line_losses_min_data["min_position"]], [mean_line_losses_min_data["loss_value"]], color="red",
                marker="D")
    plt.scatter([direction_batch_losses_min_data["min_position"]], [direction_batch_losses_min_data["loss_value"]],
                color="green",
                marker="X")

    y_lim = (min(min(min_line_losses), min_sgd_value, min_pal_value) - min(min(min_line_losses), min_sgd_value,
                                                                           min_pal_value) * 0.2, max(mean_line_losses))
    plt.plot([mean_line_losses_min_data["min_position"], mean_line_losses_min_data["min_position"]],
             [-1, mean_line_losses_min_data["loss_value"]], color="red", linewidth=linewidth,
             linestyle='dashed')

    plt.plot([direction_batch_losses_min_data["min_position"], direction_batch_losses_min_data["min_position"]],
             [-1, direction_batch_losses_min_data["loss_value"]], color="green", linewidth=linewidth,
             linestyle='dashed')
    ax = plt.gca()
    plt.xlabel("step on line")
    plt.ylabel("loss")
    ax.set_ylim(y_lim)
    plt.savefig("{0}/sgd_pal_line_{1:d}.png".format(save_path, step), dpi=200,
                bbox_inches='tight')
    plt.close()
    a = 2


def plot_line_old(sgd_update_step, mean_line_val_losses_min_data, val_line_losses, mean_line_val_losses,
                  sample_positions,
                  min_line_losses, max_line_losses, mean_line_losses, direction_batch_losses, x, y,
                  x_test, y_test, mean_line_losses_min_data, direction_batch_losses_min_data,
                  approximated_mean_loss_minimum, direction_line_approx_coefficients, fit_has_min,
                  polynomial_degree, fit_error, save_path, config_dict, step):
    linewidth = 1.5
    save_path = os.path.join(save_path, dict_to_string(config_dict))
    os.makedirs(save_path, exist_ok=True)
    matplotlib.use('Agg')
    plt.figure()
    plt.fill_between(sample_positions, min_line_losses, max_line_losses, color="blue", alpha=0.2,
                     linewidth=1)
    plt.plot(sample_positions, mean_line_losses, color="red", linewidth=linewidth, label="mean train loss")
    plt.plot(sample_positions, mean_line_val_losses, color="purple", linewidth=linewidth, label="mean val loss")
    plt.plot(sample_positions, direction_batch_losses, color="green", linewidth=linewidth, label="batch loss")

    resolution = (max(x) - min(x)) / 10000
    sgd_approx_x = batch_approx_x = np.arange(min(x), max(x), resolution)

    # sgd approx
    _0index = np.where(sample_positions == 0)[0]
    c = direction_batch_losses[_0index]
    b = (direction_batch_losses[_0index] - direction_batch_losses[_0index - 5]) / abs(
        sample_positions[_0index] - sample_positions[_0index - 5])

    a = -b / (sgd_update_step * 2)
    sgd_parabola_coefficients = [a, b, c]
    sgd_approx_y = np.polyval(sgd_parabola_coefficients, sgd_approx_x)
    plt.plot(sgd_approx_x, sgd_approx_y, color="black", linewidth=linewidth, label="sgd parabola")
    sgd_min_y_index = sgd_approx_y.argmin()

    # pal approx
    batch_approx_y = np.polyval(direction_line_approx_coefficients, batch_approx_x)
    plt.plot(batch_approx_x, batch_approx_y, color="blue", linewidth=linewidth, label="pal parabola")
    min_y_index = batch_approx_y.argmin()

    plt.scatter(sgd_approx_x[sgd_min_y_index], sgd_approx_y[sgd_min_y_index],
                color="black", marker="o", s=50, zorder=100)

    plt.scatter(batch_approx_x[min_y_index], batch_approx_y[min_y_index],
                color="blue", marker="o", s=50, zorder=100)

    plt.scatter([mean_line_losses_min_data["min_position"]], [mean_line_losses_min_data["loss_value"]], color="red",
                marker="D")
    plt.scatter([mean_line_val_losses_min_data["min_position"]], [mean_line_val_losses_min_data["loss_value"]],
                color="purple",
                marker="x")
    plt.scatter([direction_batch_losses_min_data["min_position"]], [direction_batch_losses_min_data["loss_value"]],
                color="green",
                marker="X")

    y_lim = (min(min_line_losses) - 0.1, max(mean_line_losses))

    plt.plot([sgd_approx_x[sgd_min_y_index], sgd_approx_x[sgd_min_y_index]],
             [-1, sgd_approx_y[sgd_min_y_index]], color="black", linewidth=linewidth,
             linestyle='dashed')

    plt.plot([batch_approx_x[min_y_index], batch_approx_x[min_y_index]],
             [-1, batch_approx_y[min_y_index]], color="blue", linewidth=linewidth,
             linestyle='dashed')
    plt.plot([mean_line_losses_min_data["min_position"], mean_line_losses_min_data["min_position"]],
             [-1, mean_line_losses_min_data["loss_value"]], color="red", linewidth=linewidth,
             linestyle='dashed')
    plt.plot([mean_line_val_losses_min_data["min_position"], mean_line_val_losses_min_data["min_position"]],
             [-1, mean_line_val_losses_min_data["loss_value"]], color="purple", linewidth=linewidth,
             linestyle='dashed')

    plt.plot([direction_batch_losses_min_data["min_position"], direction_batch_losses_min_data["min_position"]],
             [-1, direction_batch_losses_min_data["loss_value"]], color="green", linewidth=linewidth,
             linestyle='dashed')
    ax = plt.gca()
    plt.text(0.1, 0.9, "polynomial degree: " + str(polynomial_degree), transform=ax.transAxes)
    plt.text(0.1, 0.85, f"fit error: {fit_error:1.5f}", transform=ax.transAxes)
    plt.xlabel("step on line")
    plt.ylabel("loss")
    ax.set_ylim(y_lim)
    plt.savefig("{0}/train_line_{1:d}_deg_{2:d}.png".format(save_path, step, polynomial_degree), dpi=200,
                bbox_inches='tight')
    plt.close()
    a = 2


def get_distance_between_mins(first_min_data, second_min_data):
    distance = second_min_data["min_position"] - first_min_data["min_position"]
    return distance


def draw_losses(amount_losses_to_draw, losses, sample_positions, seed=1):
    np.random.seed(seed)
    num_sample_locations = losses.shape[1]
    num_sampled_batches_per_measure_location = losses.shape[0]
    sample_location_indexes = np.random.choice(range(num_sample_locations), amount_losses_to_draw).ravel()
    sampled_loss_indexes = np.random.choice(range(num_sampled_batches_per_measure_location),
                                            amount_losses_to_draw).ravel()
    drawn_lossses = losses[sampled_loss_indexes, sample_location_indexes]
    drawn_locations = sample_positions[sample_location_indexes]
    return drawn_lossses, drawn_locations, sample_location_indexes


def split_dataset(val_percentage, losses, losses_positions):
    validation_set_size = int(len(losses) * val_percentage / 100)
    choice = np.random.choice(range(len(losses)),
                              size=(len(losses) - validation_set_size), replace=False)
    ind = np.zeros(len(losses), dtype=bool)
    ind[choice] = True
    rest = ~ind
    x = np.array(losses_positions[ind])
    y = np.array(losses[ind])
    x_test = np.array(losses_positions[rest])
    y_test = np.array(losses[rest])
    return x, y, x_test, y_test


def batch_losses_at_each_line_location(sampled_line_losses, sample_positions, batch_size):
    if batch_size > 1:
        loss_samples_per_position = np.shape(sampled_line_losses)[
                                        0] // batch_size * batch_size  # TODO check if this fits with new data
        num_batches = loss_samples_per_position // batch_size
        num_measuring_locations = len(sample_positions)
        batched_sampled_losses = np.zeros([num_batches, num_measuring_locations])
        for location_index in range(num_measuring_locations):
            batches = sampled_line_losses[:loss_samples_per_position, location_index].reshape(-1, batch_size)
            batched_losses = np.nanmean(batches, 1)
            batched_sampled_losses[:, location_index] = batched_losses[:]

        return batched_sampled_losses
    else:
        return sampled_line_losses


def get_nearest_indexes(measured_positions, positions):
    nearest_indices = np.array([np.abs(
        measured_positions - x).argmin() for x in positions])
    return nearest_indices


def get_nearest_index(measured_positions, position):
    nearest_index = np.abs(
        measured_positions - position).argmin()
    return nearest_index


def get_best_fitting_polynomial(x, y, x_test, y_test, max_polynomial_degree):
    train_errors = []
    test_errors = []
    coefficients_list = []
    degs = []
    for polynomial_degree in range(2, max_polynomial_degree):
        coefficients = np.polyfit(x, y, polynomial_degree)
        test_error = get_mean_square_error(x_test, y_test, coefficients)
        train_error = get_mean_square_error(x, y, coefficients)
        train_errors.append(train_error)
        test_errors.append(test_error)
        degs.append(polynomial_degree)
        coefficients_list.append(coefficients)

    if len(test_errors) > 0:
        min_test = min(test_errors)
        min_test_index = test_errors.index(min_test)
        deg = degs[min_test_index]
        coefficients = coefficients_list[min_test_index]
        return coefficients, deg
    return None, None


def write_pickle(object, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(object, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sort_key(text):
    return int(text.split(".")[0].split("_")[2][:])
    # return int(text.split(".")[0][4:])


def get_step_to_derivative(coefficients, derivative=0, closest_to=0):
    # get first derivatives:
    first_deriv_coeff = np.polyder(coefficients)
    first_deriv_coeff[-1] = first_deriv_coeff[-1] - derivative
    # get roots:
    deriv_roots = np.roots(first_deriv_coeff)
    deriv_roots = deriv_roots.real[abs(deriv_roots.imag) < 1e-5]
    # get second derivative:
    if derivative == 0:
        second_deriv_coeff = np.polyder(first_deriv_coeff)
        second_deriv_root_values = np.polyval(second_deriv_coeff, deriv_roots)
        target_positions = deriv_roots[second_deriv_root_values > 0]  # minima position
    else:
        target_positions = deriv_roots
    if len(target_positions) > 0:
        abs_min_index = np.abs(target_positions - closest_to).argmin()
        # abs_min_index = np.where(np.abs(target_positions) == np.abs(target_positions).min())[0][0]
        closest_minimum = target_positions[abs_min_index]
        return closest_minimum, True
    return np.nan, False


def get_improvement(loos0, coefficients, min_position):
    return loos0 - np.polyval(coefficients, min_position)


def write_line_statistics_data(save_dict, config_dict, data_save_path):
    c2 = copy.deepcopy(config_dict)
    del c2["ori_sgd_lr"]
    config_dict_string = dict_to_string(c2)
    file_path = os.path.join(data_save_path, config_dict_string + ".pickle")
    write_pickle((save_dict, config_dict), file_path)


def check_if_line_statistics_data_exist(config_dict, data_save_path):
    config_dict_string = dict_to_string(config_dict)
    file_path = os.path.join(data_save_path, config_dict_string + ".pickle")
    return os.path.isfile(file_path)


def read_line_statistics_data(config_dict, data_save_path):
    config_dict_string = dict_to_string(config_dict)
    file_path = os.path.join(data_save_path, config_dict_string + ".pickle")
    save_dict, new_cd = load_pickle(file_path)
    return save_dict, new_cd


def dict_to_string(dict_):
    a = [str(key) + "_" + str(value) for key, value in zip(dict_.keys(), dict_.values())]
    b = "_".join(a)
    return b


def dict_product(inp):
    return (dict(zip(inp.keys(), values)) for values in itertools.product(*inp.values()))
