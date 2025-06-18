import os
from collections import defaultdict
from datetime import datetime, timedelta
from moviepy.editor import VideoFileClip
from statistics import mean, median
import sys
import json
import socket

def get_video_duration(video_path):
    """Get the duration of a video file in seconds."""
    try:
        with VideoFileClip(video_path) as video:
            return video.duration
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return 0  # Return 0 if video can't be read

def calculate_task_times_with_stats(folder_path):
    # Dictionary to store task data: {task_name: [(folder, start_time, end_time, duration)]}
    task_data = defaultdict(list)
    
    # Walk through the folder
    for root, dirs, files in os.walk(folder_path):
        # Check if the current folder is a 7-digit task folder
        folder_name = os.path.basename(root)
        if folder_name.isdigit() and len(folder_name) == 7:
            # Get the parent folder as the task name
            task_name = os.path.basename(os.path.dirname(root))
            # Get folder creation time
            stat = os.stat(root)
            try:
                # Try to get creation time (Windows)
                created_time = stat.st_ctime
            except AttributeError:
                # Fallback to last modification time (Unix)
                created_time = stat.st_mtime
            
            start_time = datetime.fromtimestamp(created_time)
            
            # Path to the demo.mp4 video
            video_path = os.path.join(root, "observation", "cam_left_wrist", "color_image", "demo.mp4")
            
            # Get video duration
            duration_seconds = 0
            if os.path.exists(video_path):
                duration_seconds = get_video_duration(video_path)
            else:
                print(f"Video not found: {video_path}")
            
            # Calculate end time
            end_time = start_time + timedelta(seconds=duration_seconds)
            
            # Store the data
            task_data[task_name].append((folder_name, start_time, end_time, duration_seconds))
    
    # Process results for each task
    results = {}
    for task_name, entries in task_data.items():
        # Sort entries by start time
        entries.sort(key=lambda x: x[1])
        
        # Calculate intervals between consecutive entries
        intervals = []
        for i in range(1, len(entries)):
            prev_end = entries[i-1][2]  # End time of previous entry
            curr_start = entries[i][1]  # Start time of current entry
            interval = (curr_start - prev_end).total_seconds()  # Interval in seconds
            intervals.append({
                'from_folder': entries[i-1][0],
                'to_folder': entries[i][0],
                'interval_seconds': interval
            })
        
        # Calculate duration and interval statistics
        durations = [entry[3] for entry in entries]  # Duration in seconds
        duration_mean = mean(durations) if durations else 0
        duration_median = median(durations) if durations else 0
        interval_mean = mean([i['interval_seconds'] for i in intervals]) if intervals else 0
        interval_median = median([i['interval_seconds'] for i in intervals]) if intervals else 0
        
        # Calculate the new combined statistics
        duration_interval_mean = duration_mean + interval_mean
        duration_interval_median = duration_median + interval_median
        
        results[task_name] = {
            'entries': [
                {
                    'folder': entry[0],
                    'start_time': entry[1].strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': entry[2].strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_seconds': entry[3]
                } for entry in entries
            ],
            'intervals': intervals,
            'stats': {
                'duration_mean_seconds': duration_mean,
                'duration_median_seconds': duration_median,
                'interval_mean_seconds': interval_mean,
                'interval_median_seconds': interval_median,
                'duration_interval_mean': duration_interval_mean,
                'duration_interval_median': duration_interval_median
            }
        }
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python calculate_task_times_with_stats.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    device_id = sys.argv[2]
    model = sys.argv[3]
    save_path = os.path.join('data_statistics',model)

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    results = calculate_task_times_with_stats(folder_path)
    
    print("Task Timing and Statistics:")
    print("=" * 50)
    for task_name, data in results.items():
        print(f"\nTask: {task_name}")
        print("Entries:")
        for entry in data['entries']:
            print(f"  Folder: {entry['folder']}")
            print(f"    Start Time: {entry['start_time']}")
            print(f"    End Time: {entry['end_time']}")
            print(f"    Duration: {entry['duration_seconds']:.2f} seconds")
        
        if data['intervals']:
            print("\n  Intervals between consecutive tasks:")
            for interval in data['intervals']:
                print(f"    From {interval['from_folder']} to {interval['to_folder']}: "
                      f"{interval['interval_seconds']:.2f} seconds")
        
        print("\n  Statistics:")
        print(f"    Mean Video Duration: {data['stats']['duration_mean_seconds']:.2f} seconds")
        print(f"    Median Video Duration: {data['stats']['duration_median_seconds']:.2f} seconds")
        print(f"    Mean Interval: {data['stats']['interval_mean_seconds']:.2f} seconds")
        print(f"    Median Interval: {data['stats']['interval_median_seconds']:.2f} seconds")
        print(f"    Duration + Interval Mean: {data['stats']['duration_interval_mean']:.2f} seconds")
        print(f"    Duration + Interval Median: {data['stats']['duration_interval_median']:.2f} seconds")
        
        print("-" * 50)
    
    print("\nSummary:")
    print("=" * 50)
    total_tasks = sum(len(data['entries']) for data in results.values())
    print(f"Total tasks processed: {total_tasks}")
    print(f"Total task types: {len(results)}")

    # Get current date in YYYYMMDD format and last digit of IP
    current_date = datetime.now().strftime("%Y%m%d")
    
    # Save results to JSON file
    output_filename = f"all_result_{current_date}_{model}_{device_id}.json"
    output_path = os.path.join(save_path, output_filename)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")