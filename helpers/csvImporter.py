import csv
import sys


class CSVImporter:

    def import_training_file(self, filename):
        print("Importing training file...")

        file_content = []

        with open(filename) as csvfile:
            read_csv = csv.reader(csvfile, delimiter=",")
            next(read_csv, None)  # Skip header

            for row in read_csv:
                for i in range(1,len(row)):
                    if(int(row[i])>0):
                        row[i]= 1 
                    else:
                        row[i]=0
                cur_data = {
                    "input": row[1:],
                    "output": self.__transform_to_binary(row[0])
                }
                file_content.append(cur_data)

        print("Training file imported!")
        return file_content

    def import_test_file(self, filename):
        print("Importing test file...")

        file_content = []

        with open(filename) as csvfile:
            read_csv = csv.reader(csvfile, delimiter=",")
            next(read_csv, None)  # Skip header

            for row in read_csv:
                cur_data = {
                    "input": row[0:],
                }
                file_content.extend(cur_data)

        print("Test file imported!")
        return file_content

    def __transform_to_binary(self, number):
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result[int(number)] = 1
        return result



