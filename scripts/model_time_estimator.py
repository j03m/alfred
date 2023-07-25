def estimate_time(_model: PPO):
    start_time = time.time()
    _model.learn(total_timesteps=100)
    end_time = time.time()
    _time_per_episode = (end_time - start_time) * 100
    return _time_per_episode


def convert_seconds_to_time(estimated_time: int):
    # Get a timedelta object representing the duration
    timedelta = datetime.timedelta(seconds=estimated_time)
    return str(timedelta)

time_per_episode = estimate_time(model)  # Time for 100 episodes
final_time = time_per_episode * args.training_intervals
print(f"Estimated time for {args.training_intervals} episodes: {convert_seconds_to_time(final_time)}")