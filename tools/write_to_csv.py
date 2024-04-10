import csv


def write_to_csv(data, file_path):
    try:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in zip(*data):
                writer.writerow(row)
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"Error occurred while writing to CSV: {e}")
