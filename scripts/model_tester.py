
if eval:

    print("Agent status, before training")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.benchmark_intervals)
    print(f"(pre train) mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"(pre train profit) {env}")

if test:
    obs, data = env.reset()
    done = False
    state = None
    while not done:
        action, state = model.predict(obs, state, episode_start=False)
        # take the action and observe the next state and reward
        obs, reward, _, done, info_ = env.step(action)
    env.ledger.to_csv(f"./backtests/{args.symbol}-model-back-test.csv")
    print(f"(post test profit) {env}")