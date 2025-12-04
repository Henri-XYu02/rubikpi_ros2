#!/usr/bin/env python3
"""
Standalone script to calculate error between estimated and ground truth landmark positions.
Run this after your EKF-SLAM completes and prints the final landmark map.
"""

import numpy as np
import math

# Ground truth landmark positions
ground_truth = {
    0: {'x': 2.5, 'y': 0.2},    # Right wall, bottom
    1: {'x': 0.0, 'y': 2.3},    # Left wall, top
    2: {'x': 2.5, 'y': 2.3},    # Right wall, top
    3: {'x': 2.3, 'y': 2.5},    # Top wall, right
    4: {'x': 1.2, 'y': 2.5},    # Top wall, middle
    5: {'x': 0.2, 'y': 2.5},    # Top wall, left
    6: {'x': 0.0, 'y': 1.3},    # Left wall, middle
    7: {'x': 0.0, 'y': 0.2},    # Left wall, bottom
    8: {'x': 2.5, 'y': 1.3},    # Right wall, middle
    9: {'x': 2.3, 'y': 0.0},    # Bottom wall, right
    10: {'x': 1.3, 'y': 0.0},   # Bottom wall, middle
    11: {'x': 0.2, 'y': 0.0},   # Bottom wall, left
}

# Estimated positions from EKF-SLAM
# YOU NEED TO UPDATE THESE VALUES FROM YOUR ROBOT'S OUTPUT
# Copy the final landmark positions from your terminal output

# square multiple times
estimated = {
    # Format: tag_id: {'x': estimated_x, 'y': estimated_y}
    # Example (replace with your actual values):
    # 0: {'x': 1.98, 'y': 0.02},
    # 1: {'x': -0.04, 'y': -0.58},
    # Add your estimated values here...
    0: {'x': 2.41, 'y':0.24},
    1: {'x': -0.06, 'y':2.22},
    2: {'x': 2.15, 'y':2.41},
    3: {'x': 2.09, 'y':2.73},
    4: {'x': 1.10, 'y':2.90},
    5: {'x': 0.10, 'y':2.65},
    6: {'x': 0.13, 'y':1.42},
    7: {'x': 0.39, 'y':0.37},
    8: {'x': 2.19, 'y':1.41},
    9: {'x': 2.52, 'y':-0.07},
    10: {'x': 1.73, 'y':-0.23},
    11: {'x': 0.56, 'y':0.01},
}


def calculate_errors(ground_truth, estimated):
    """
    Calculate position errors for each landmark.
    
    Returns:
    --------
    errors : dict
        Dictionary with tag_id as key and error metrics as value
    """
    errors = {}
    
    for tag_id in ground_truth.keys():
        if tag_id not in estimated:
            print(f"Warning: Tag {tag_id} not found in estimated positions (not detected)")
            continue
        
        gt = ground_truth[tag_id]
        est = estimated[tag_id]
        
        # Calculate error in x and y
        error_x = est['x'] - gt['x']
        error_y = est['y'] - gt['y']
        
        # Calculate Euclidean distance error
        euclidean_error = math.sqrt(error_x**2 + error_y**2)
        
        errors[tag_id] = {
            'error_x': error_x,
            'error_y': error_y,
            'euclidean': euclidean_error,
            'gt_x': gt['x'],
            'gt_y': gt['y'],
            'est_x': est['x'],
            'est_y': est['y']
        }
    
    return errors

def print_results(errors):
    """Print detailed error analysis"""
    print("="*80)
    print("LANDMARK POSITION ERROR ANALYSIS")
    print("="*80)
    print()
    
    # Detailed per-landmark errors
    print("Per-Landmark Errors:")
    print("-"*80)
    print(f"{'Tag':<5} {'GT Position':<20} {'Est Position':<20} {'Error (X, Y)':<20} {'Euclidean':<10}")
    print("-"*80)
    
    for tag_id, error in sorted(errors.items()):
        gt_pos = f"({error['gt_x']:6.2f}, {error['gt_y']:6.2f})"
        est_pos = f"({error['est_x']:6.2f}, {error['est_y']:6.2f})"
        err_xy = f"({error['error_x']:6.3f}, {error['error_y']:6.3f})"
        
        print(f"{tag_id:<5} {gt_pos:<20} {est_pos:<20} {err_xy:<20} {error['euclidean']:6.3f}m")
    
    print("-"*80)
    print()
    
    # Summary statistics
    euclidean_errors = [e['euclidean'] for e in errors.values()]
    error_x_values = [e['error_x'] for e in errors.values()]
    error_y_values = [e['error_y'] for e in errors.values()]
    
    print("Summary Statistics:")
    print("-"*80)
    print(f"Number of landmarks evaluated: {len(errors)}")
    print(f"Number of landmarks not detected: {len(ground_truth) - len(errors)}")
    print()
    print(f"Average Euclidean Error:  {np.mean(euclidean_errors):.4f} m")
    print(f"Median Euclidean Error:   {np.median(euclidean_errors):.4f} m")
    print(f"Std Dev Euclidean Error:  {np.std(euclidean_errors):.4f} m")
    print(f"Min Euclidean Error:      {np.min(euclidean_errors):.4f} m")
    print(f"Max Euclidean Error:      {np.max(euclidean_errors):.4f} m")
    print()
    print(f"Average X Error:          {np.mean(error_x_values):.4f} m")
    print(f"Average Y Error:          {np.mean(error_y_values):.4f} m")
    print(f"RMS X Error:              {np.sqrt(np.mean(np.array(error_x_values)**2)):.4f} m")
    print(f"RMS Y Error:              {np.sqrt(np.mean(np.array(error_y_values)**2)):.4f} m")
    print(f"RMS Total Error:          {np.sqrt(np.mean(np.array(euclidean_errors)**2)):.4f} m")
    print("="*80)

def parse_from_terminal_output(terminal_text):
    """
    Parse estimated positions from terminal output.
    
    Example terminal output:
        tag_0: pos=(1.980, 0.020), std=(0.150, 0.145)
        tag_1: pos=(-0.045, -0.585), std=(0.120, 0.130)
    
    Parameters:
    -----------
    terminal_text : str
        Multi-line string containing the terminal output
    
    Returns:
    --------
    estimated : dict
        Dictionary of estimated positions
    """
    estimated = {}
    
    for line in terminal_text.strip().split('\n'):
        if 'tag_' in line and 'pos=' in line:
            # Extract tag ID
            tag_id = int(line.split('tag_')[1].split(':')[0])
            
            # Extract position
            pos_str = line.split('pos=(')[1].split(')')[0]
            x_str, y_str = pos_str.split(',')
            x = float(x_str.strip())
            y = float(y_str.strip())
            
            estimated[tag_id] = {'x': x, 'y': y}
    
    return estimated

def main():
    """Main function"""
    
    # Check if estimated positions are filled in
    if len(estimated) == 0:
        print("="*80)
        print("ERROR: No estimated positions provided!")
        print("="*80)
        print()
        print("Please update the 'estimated' dictionary in this script with your")
        print("EKF-SLAM results, or paste your terminal output below.")
        print()
        print("Option 1: Manual entry")
        print("-" * 40)
        print("Update the 'estimated' dictionary in the script like this:")
        print()
        print("estimated = {")
        print("    0: {'x': 1.98, 'y': 0.02},")
        print("    1: {'x': -0.04, 'y': -0.58},")
        print("    # ... add all detected tags")
        print("}")
        print()
        print("Option 2: Paste terminal output")
        print("-" * 40)
        print("Or uncomment and use the terminal_text example below:")
        print()
        print("# Example:")
        print('# terminal_text = """')
        print('#   tag_0: pos=(1.980, 0.020), std=(0.150, 0.145)')
        print('#   tag_1: pos=(-0.045, -0.585), std=(0.120, 0.130)')
        print('# """')
        print('# estimated = parse_from_terminal_output(terminal_text)')
        print()
        print("="*80)
        return
    
    # Calculate errors
    errors = calculate_errors(ground_truth, estimated)
    
    # Print results
    print_results(errors)
    
    # Check for undetected landmarks
    undetected = set(ground_truth.keys()) - set(estimated.keys())
    if undetected:
        print()
        print("WARNING: The following tags were not detected:")
        for tag_id in sorted(undetected):
            gt = ground_truth[tag_id]
            print(f"  Tag {tag_id}: GT position ({gt['x']:.2f}, {gt['y']:.2f})")

if __name__ == '__main__':
    print()
    print("Ground Truth Landmark Positions:")
    print("-"*40)
    for tag_id, pos in sorted(ground_truth.items()):
        print(f"Tag {tag_id}: ({pos['x']:6.2f}, {pos['y']:6.2f})")
    print()
    
    # OPTION 1: Manual entry in 'estimated' dictionary above
    
    # OPTION 2: Paste your terminal output here (uncomment to use)
    # Example terminal output from your EKF-SLAM:
    """
    terminal_text = '''
    tag_0: pos=(1.980, 0.020), std=(0.150, 0.145)
    tag_1: pos=(-0.045, -0.585), std=(0.120, 0.130)
    tag_2: pos=(1.340, 1.485), std=(0.180, 0.175)
    tag_3: pos=(-0.885, 0.095), std=(0.165, 0.155)
    tag_4: pos=(-0.495, 0.965), std=(0.145, 0.160)
    tag_5: pos=(0.005, 1.495), std=(0.170, 0.165)
    tag_7: pos=(1.095, -0.595), std=(0.155, 0.150)
    tag_8: pos=(1.995, 0.895), std=(0.160, 0.155)
    '''
    estimated = parse_from_terminal_output(terminal_text)
    """
    
    main()