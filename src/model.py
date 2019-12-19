import pandas as pd
import numpy as np
import os
import sys


# Maintains all of the data and handles the data manipulation for this
# application
class DataModel:
    def __init__(self):
        self.input_fname = None
        self.output_fname = None
        self.inpput_df = None
        self.output_df = None
        self.display_df = None
        self.save_df = None
        self.current_row_index = None
        self.num_notes = None
        self.annotation_key = 'ANNOTATION'

    def rpdr_to_csv(self):
        csvfile = ''.join(
            self.input_fname.split('.')[
                :-1]) + '.csv'
        # corresponding CSV already exist
        if os.path.isfile(csvfile) and ('report_text' in pd.read_csv(
                csvfile).columns.values.tolist() or 'comments' in pd.read_csv(
                csvfile).columns.values.tolist()):
            self.input_fname = ''.join(
                self.input_fname.split('.')[:-1]) + '.csv'
        # reformat RPDR to CSV file
        else:
            with open(self.input_fname, 'r') as file:
                data, header, fields = [], [], []
                for line in file:
                    line = line.rstrip()
                    if line.strip() == '':
                        continue
                    if not header:
                        if line.count('|') < 8:
                            messagebox.showerror(
                                title="Error",
                                message="Something went wrong, did you select an appropriately formatted RPDR file to perform the Regex on?")
                            return
                        header = [field.lower().strip()
                                  for field in line.split('|')]
                        continue
                    if not fields and '|' in line:
                        fields = [field.lower().strip()
                                  for field in line.split('|')]
                        fields[-1] = line
                        report = []
                    elif 'report_end' in line:
                        report.append(line)
                        fields[-1] += '\n'.join(report)
                        data.append(fields)
                        fields = []
                    else:
                        report.append(line)
            data = pd.DataFrame(data, columns=header)
            self.input_fname = ''.join(
                self.input_fname.split('.')[:-1]) + '.csv'
            data.to_csv(self.input_fname, index=False)

    def write_to_annotation(self):
        if self.save_df is None or self.current_row_index is None:
            return
        pathname = os.path.dirname(sys.argv[0])
        self.save_df.to_csv(
            os.path.join(
                pathname,
                self.output_fname),
            index=False)

    def get_annotation(self):
        if self.annotation_key in self.output_df:
            current_row_index = self.display_df.index[self.current_row_index]
            val = self.output_df.at[current_row_index, self.annotation_key]
            if val is not None and not pd.isnull(val):
                try:
                    return int(float(val))
                except BaseException:
                    return val
        return ''
