import os
import vessl

from cfg import get_cfg
from torch.utils.tensorboard import SummaryWriter
from agent.ppo import *
from environment.env import *


if __name__ == "__main__":
    cfg = get_cfg()
    vessl.init(organization="snu-eng-dgx", project="nesting", hp=cfg)

    lr = cfg.lr
    gamma = cfg.gamma
    lmbda = cfg.lmbda
    eps_clip = cfg.eps_clip
    K_epoch = cfg.K_epoch
    T_horizon = cfg.T_horizon

    model_dir = '/output/train/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = '/output/train/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    simulation_dir = '/output/train/simulation/'
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    env = HiNEST(look_ahead=5)
    agent = Agent(env.state_size, env.x_action_size, env.y_action_size, env.a_action_size, lr, gamma, lmbda, eps_clip, K_epoch)
    writer = SummaryWriter(log_dir)

    if cfg.load_model:
        checkpoint = torch.load(cfg.model_path)
        start_episode = checkpoint['episode'] + 1
        agent.network.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_episode = 1

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, loss\n')

    for e in range(start_episode, cfg.n_episode + 1):
        # vessl.log(payload={"Train/learnig_rate": agent.scheduler.get_last_lr()[0]}, step=e)
        # writer.add_scalar("Training/Learning Rate", agent.scheduler.get_last_lr()[0], e)

        s = env.reset()
        update_step = 0
        r_epi = 0.0
        avg_loss = 0.0
        done = False

        while not done:
            possible_actions = env.get_possible_actions()
            a, prob, mask = agent.get_action(s, possible_actions)
            s_prime, r, efficiency, done = env.step(a)

            agent.put_data((s, a[0], a[1], a[2], r, s_prime, prob[0], prob[1], prob[2], mask, done))
            s = s_prime

            r_epi += r

            update_step += 1

            if done:
                vessl.log(step=e, payload={'reward': r_epi})
                vessl.log(step=e, payload={'efficiency': efficiency})
                break
            avg_loss += agent.train()

        # agent.scheduler.step()

        # print("episode: %d | reward: %.2f | distance : %d | loss : %.2f" % (e, r_epi, env.model.crane_dis_cum["Crane1"], avg_loss))
        with open(log_dir + "train_log.csv", 'a') as f:
            f.write('%d,%1.2f,1.2%f\n' % (e, r_epi, avg_loss))

        writer.add_scalar("Training/Reward", r_epi, e)
        writer.add_scalar("Training/Loss", avg_loss / update_step, e)
        # for i in env.model.crane_dis_cum.keys():
        #     writer.add_scalar("Performance/Distance", env.model.crane_dis_cum[i], e)

        # if e % 20 == 0:
        #     with torch.no_grad():
        #         delay, move, priority_ratio = [], [], []
        #         for path in validation_path:
        #             df_ship_test = pd.read_excel(path, sheet_name="ship", engine='openpyxl')
        #             df_initial_test = pd.read_excel(path, sheet_name="initial", engine='openpyxl')
        #             test_env = QuayScheduling(df_quay, df_ship_test, df_initial_test, w_delay, w_move, w_priority, rand=False)
        #
        #             s = test_env.reset()
        #             done = False
        #
        #             while not done:
        #                 possible_actions = test_env.get_possible_actions()
        #                 a, prob, mask = agent.get_action(s, possible_actions)
        #                 s_prime, r, done = test_env.step(a)
        #                 s = s_prime
        #
        #                 if done:
        #                     log = test_env.get_logs()
        #                     break
        #
        #             average_delay = calculate_average_delay(log)
        #             move_ratio = calculate_move_ratio(log)
        #             priority_ratio = calculate_priority_ratio(log)
        #
        #             name = path.split("/")[-1][:-5] + "_log.csv"
        #             with open(log_dir + name, 'a') as f:
        #                 f.write('%d,%1.4f, %1.4f, %1.4f\n' % (e, average_delay, move_ratio, priority_ratio))
        #
        #             writer.add_scalars("Validation/Average Delay", {name.split("_")[1]: average_delay}, e)
        #             writer.add_scalars("Validation/Move Ratio", {name.split("_")[1]: move_ratio}, e)
        #             writer.add_scalars("Validation/Priority Ratio", {name.split("_")[1]: priority_ratio}, e)


        if e % 1000 == 0:
            agent.save_network(e, model_dir)
            # env.model.mointor.save(simulation_dir + "episode%d.csv" % e)

    writer.close()