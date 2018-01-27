import csv
import numpy as np

class CSVImporter:

    def import_file(self, filename):
        print("Importing training file...")

        file_content = []

        with open(filename) as csvfile:
            read_csv = csv.reader(csvfile, delimiter=",")
            next(read_csv, None)  # Skip header

            for row in read_csv:

                cur_data = []

                # Normalize Inputs
                for i in range(1, len(row)):
                    cur_data.append(self.__normalize(float(row[i])))

                # Add label
                cur_data.extend(self.__transform_to_binary(row[0]))

                file_content.append(cur_data)

        print("Training file imported!")
        return file_content

    @staticmethod
    def __normalize(value):
        return float((value - 128) / 128)

    @staticmethod
    def __transform_to_binary(number):
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result[int(number)] = 1
        return result



