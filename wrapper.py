import csv
from subprocess import call

# Creates two csv files and returns the filename
def split_csv_into_numeric_and_text(csvfile):
	data = list(csv.reader(open(csvfile)))
	data = data[1:]
	
	num_data = []
	text_data = []

	for row in data:
		text_data.append([row[-1]])
		num_data.append(row[:-1])

	with open("pyout_text_data.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(text_data)

	with open("pyout_num_data.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(num_data)

	return "pyout_text_data.csv", "pyout_num_data.csv"

def call_ga_script(text_file, num_file):
	

def cleanup():
	call(["rm", "num_data.csv"])
	call(["rm", "text_data.csv"])	

if __name__ == "__main__":

	text_file, num_file = split_csv_into_numeric_and_text("./lung_info.csv")
	call_ga_script(text_file, num_file)

	cleanup()