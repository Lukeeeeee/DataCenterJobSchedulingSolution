class Config():
    # simulation
    T = 2200
    sim_t = 2000 + 1
    current_time = 0

    # Job
    process_t_lower = 1
    process_t_upper = 5
    job_resource_lower = 5
    job_resource_upper = 15

    # Server
    server_r = 40
    server_count = 6
    server_heat_constant = 200

    server_pos_x = [0, 1, -2, -2, 3, 3]
    server_pos_y = [0, 0, 0, 2, 1, 3]

    a = 0.5
    b = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # Reward
    # alpha = 10
    alpha = 0.5
    # beta = 0.2
    beta = 0.5

    # Q net in out put
    server_state_dim = 7
    total_server_state_dim = server_count * server_state_dim
    server_feature_dim = 2
    job_state_dim = 5
    dc_state_dim = 1
    action_dim = server_count

    # Q net size
    server_feature_layer1_size = 10
    q_net_layer1_size = 20
    q_net_layer2_size = 10

    # TRAIN PARAMETERS
    gamma = 0.8
    learning_rate = 0.3
    batch_size = 500
    sample_count = sim_t - 1
    batch_iter = sample_count / batch_size
    epoch = 1000
    epsilon = 0.9
    update_target_q_every_iter = batch_iter / 4
    save_data_every_epoch = 5
