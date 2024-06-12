import time
import numpy as np
import csv

def compute_distance_matrix(feature_indices, data):
    feature_values = data[:, feature_indices]
    expanded_features = feature_values[:, np.newaxis, :]
    squared_diff_matrix = np.sum((expanded_features - feature_values[np.newaxis, :, :]) ** 2, axis=2)
    return squared_diff_matrix

def leave_one_out_validation(feature_indices, data, target_labels):
    distance_matrix = compute_distance_matrix(feature_indices, data)
    np.fill_diagonal(distance_matrix, np.inf)
    nearest_indices = np.argmin(distance_matrix, axis=1)
    predicted_labels = target_labels[nearest_indices]
    match_count = np.sum(predicted_labels == target_labels)
    accuracy_score = match_count / len(target_labels)
    return accuracy_score

def forward_selection_optimized(dataset):
    total_features = list(range(1, dataset.shape[1]))
    selected_features = []
    highest_accuracy = 0
    optimal_feature_set = []
    labels = dataset[:, 0]

    all_features_accuracy = leave_one_out_validation(total_features, dataset, labels)
    print(f"Running nearest neighbor with all features using 'leaving-one-out' evaluation gives accuracy: {all_features_accuracy * 100:.1f}%\n")


    with open("./forward_selection.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Feature Set', 'Accuracy'])
        initial_accuracy = leave_one_out_validation([], dataset, labels)
        feature_set_str = f"{{{', '.join(map(str, selected_features))}}}"
        csvwriter.writerow([feature_set_str, round(initial_accuracy * 100, 2)])

        while total_features:
            current_best_accuracy = 0
            best_feature = None

            for feature in total_features:
                temp_selected_features = selected_features + [feature]
                accuracy = leave_one_out_validation(temp_selected_features, dataset, labels)

                print(f"\tUsing feature(s) {temp_selected_features}, accuracy is {accuracy * 100:.1f}%")

                if accuracy > current_best_accuracy:
                    current_best_accuracy = accuracy
                    best_feature = feature

            if best_feature is not None:
                selected_features.append(best_feature)
                total_features.remove(best_feature)
                
                print(f"Feature set: {selected_features} was best, accuracy: {current_best_accuracy * 100:.1f}%")

                print(f"Selected {len(selected_features)} out of {len(selected_features) + len(total_features)} features.\n")

                feature_set_str = f"{{{', '.join(map(str, selected_features))}}}"
                csvwriter.writerow([feature_set_str, round(current_best_accuracy * 100, 2)])
                
                if current_best_accuracy > highest_accuracy:
                    highest_accuracy = current_best_accuracy
                    optimal_feature_set = selected_features.copy()
                else:
                    print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n")

    # print(f"Best feature subset: {optimal_feature_set}, accuracy: {highest_accuracy * 100:.1f}%")
    return optimal_feature_set, highest_accuracy

def backward_elimination_optimized(dataset):
    total_features = list(range(1, dataset.shape[1]))
    selected_features = total_features.copy()
    highest_accuracy = 0
    optimal_feature_set = selected_features.copy()
    labels = dataset[:, 0]

    initial_accuracy = leave_one_out_validation(selected_features, dataset, labels)
    print(f"Running nearest neighbor with all features using 'leaving-one-out' evaluation gives accuracy: {initial_accuracy * 100:.1f}%\n")
    highest_accuracy = initial_accuracy
    
    with open("./backward_selection.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Feature Set', 'Accuracy'])
        feature_set_str = f"{{{', '.join(map(str, selected_features))}}}"
        csvwriter.writerow([feature_set_str, round(highest_accuracy * 100, 2)])
        while selected_features:
            current_best_accuracy = 0
            worst_feature = None

            for feature in selected_features:
                temp_selected_features = selected_features.copy()
                temp_selected_features.remove(feature)
                accuracy = leave_one_out_validation(temp_selected_features, dataset, labels)

                print(f"\tUsing feature(s) {temp_selected_features}, accuracy is {accuracy * 100:.1f}%")

                if accuracy > current_best_accuracy:
                    current_best_accuracy = accuracy
                    worst_feature = feature

            if worst_feature is not None:
                selected_features.remove(worst_feature)

                print(f"Feature set: {selected_features} was best, accuracy: {current_best_accuracy * 100:.1f}%")

                print(f"Removed {worst_feature}. Remaining {len(selected_features)} features.\n")
                
                feature_set_str = f"{{{', '.join(map(str, selected_features))}}}"
                csvwriter.writerow([feature_set_str, round(current_best_accuracy * 100, 2)])

                if current_best_accuracy > highest_accuracy:
                    highest_accuracy = current_best_accuracy
                    optimal_feature_set = selected_features.copy()
                else:
                    print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n")

    print(f"Best feature subset using: {optimal_feature_set}, accuracy: {highest_accuracy * 100:.1f}%")
    return optimal_feature_set, highest_accuracy

def main():
    print("Welcome to Prathik's Feature Selection Algorithm.")
    file = input("Type in the name of the file to test: ")
    dataset = np.loadtxt(file)
    num_features = dataset.shape[1] - 1
    num_instances = dataset.shape[0]
    
    print(f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")
    
    while True:
        try:
            print("Choose the search algorithm: 1 - Forward Selection, 2 - Backward Elimination")
            choice = int(input().strip())
            if choice not in [1, 2]:
                raise ValueError("Invalid choice, please select 1 or 2.")
            break
        except ValueError as e:
            print(e)
    
    start_time = time.time()

    if choice == 1:
        print("Starting Forward Selection...")
        best_features, best_accuracy = forward_selection_optimized(dataset)
    elif choice == 2:
        print("Starting Backward Elimination...")
        best_features, best_accuracy = backward_elimination_optimized(dataset)

    elapsed_time = time.time() - start_time

    print(f"Finished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy * 100:.1f}%")
    print(f"Elapsed time: {elapsed_time:.1f} seconds")

if __name__ == "__main__":
    main()
