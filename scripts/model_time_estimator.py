time_per_episode = estimate_time(model)  # Time for 100 episodes
final_time = time_per_episode * args.training_intervals
print(f"Estimated time for {args.training_intervals} episodes: {convert_seconds_to_time(final_time)}")