import tkinter as tk
from tkinter import font, filedialog, messagebox
from src.model import DataModel
from src.extract_values import run_regex
import pandas as pd
import ast
import re
import os


class MainApplication(tk.Frame):
    def __init__(self, master):

        tk.Frame.__init__(self, master)
        self.master = master
        self.setup_interface(master)
        self.data_model = DataModel()
        self.positivehit = False
        self.patientlevel = False

    # Set up button click methods
    def on_select_file(self):
        self.disable_button()
        self.load_annotation = False
        self.data_model = DataModel()

        file = filedialog.askopenfilename(title="Select File")
        if file:
            self.data_model.input_fname = file
            self.file_text.config(
                text=self.data_model.input_fname.split('/')[-1])
        else:
            messagebox.showerror(
                title="Error",
                message="Something went wrong, did you select an appropriately file?")
            return
        f = self.data_model.input_fname.split('.')[-1]
        # RPDR to CSV
        if f == 'txt': self.data_model.rpdr_to_csv()

        self.regex_button.config(state='normal')
        self.load_button.config(state='normal')

        # set the default value of note and id key
        if f in ['csv', 'xls', 'xlsx', 'txt']:
            self.note_key_entry = tk.StringVar(self.right_options_frame)
            self.patient_id_entry = tk.StringVar(self.right_options_frame)
            if 'xls' in f:
                OPTIONS = pd.read_excel(
                    self.data_model.input_fname).columns.values.tolist()
            else:
                OPTIONS = pd.read_csv(
                    self.data_model.input_fname).columns.values.tolist()
            self.patient_id_entry.set(OPTIONS[0])
            self.note_key_entry.set(OPTIONS[1])
            # note ID column
            if 'empi' in OPTIONS:
                self.patient_id_entry.set('empi')
            else:
                for o in OPTIONS:
                    if 'id' in o.lower(): 
                        self.patient_id_entry.set(o); break
            # report text column
            if 'report_text' in OPTIONS:
                self.note_key_entry.set('report_text')
            elif 'comments' in OPTIONS:
                self.note_key_entry.set('comments')
            else:
                for o in OPTIONS:
                    if 'text' in o.lower(): 
                        self.note_key_entry.set(o); break
            # drop down menu
            try:
                self.note_key_entry_menu = tk.OptionMenu(
                    self.right_options_frame, self.note_key_entry, *OPTIONS)
                self.note_key_entry_menu.grid(column=1, row=3, sticky='we')
                self.note_key_entry_menu.config(
                    font=font.Font(family='Roboto', size=12))
                self.patient_id_entry_menu = tk.OptionMenu(
                    self.right_options_frame, self.patient_id_entry, *OPTIONS)
                self.patient_id_entry_menu.grid(column=1, row=2, sticky='we')
                self.patient_id_entry_menu.config(
                    font=font.Font(family='Roboto', size=12))
            except BaseException:
                messagebox.showerror(
                    title="Error",
                    message="Something went wrong, did you select an appropriately file to perform the Regex on?")
                return
        else:
            messagebox.showerror(
                title="Error",
                message="Something went wrong, did you select an appropriately formatted RPDR or CSV file to perform the Regex on?")
            return


    def on_run_regex(self):
        if not self.data_model.input_fname:
            messagebox.showerror(
                title="Error",
                message="Please select an input file using the 'Select File' button.")
            return
        # retrieve phrases
        self.phrases = {}
        for i in range(1, 4):
            self.phrases[i] = self.regex_text[i].get(1.0, 'end-1c').strip()
            self.label_name[i] = self.label_text[i].get(1.0, 'end-1c').strip()

        if self.phrases[1] == self.original_regex_text or len(
                self.phrases[1]) == 0:
            messagebox.showerror(
                title="Error",
                message="Please input comma-separated phrases to search for. ")
            return
        # get the note and id key from CSV or RPDR file
        self.note_key = self.note_key_entry.get()
        self.patient_key = self.patient_id_entry.get()
        if not self.note_key:
            messagebox.showerror(
                title='Error',
                message='Please input the column name for notes.')
            return
        if not self.patient_key:
            messagebox.showerror(
                title='Error',
                message='Please input the column name for note IDs.')
            return
        self.enable_button()
        self.load_button.config(state='disabled')


    def on_load_annotation(self):
        if not self.data_model.input_fname or '.csv' not in self.data_model.input_fname:
            messagebox.showerror(
                title="Error",
                message="Please select an input file using the 'Select File' button.")
            return
        self.phrases = {}
        self.load_annotation = True
        self.regex_button.config(state='disabled')
        self.data_model.output_df = pd.read_csv(
            self.data_model.input_fname)
        columns = self.data_model.output_df.columns.values.tolist()
        if 'L1_' not in columns[2] or 'L1_' not in columns[3] or 'L1_' not in columns[4] or 'K1_' not in columns[5]:
            messagebox.showerror(
                title="Error",
                message="Something went wrong, did you select an appropriately output CSV file?")
            return
        self.patient_key, self.note_key = columns[:2]
        self.phrases[2] = self.phrases[3] = self.original_regex_text

        try:
            l, idx = 1, 2
            while idx+3<len(columns) and l < 4 and 'L%d_'%l in columns[idx] and 'K%d_'%l in columns[idx+3]:
                self.label_name[l] = columns[idx][3:]
                self.phrases[l] = columns[idx+3][3:]
                l += 1; idx += 4
        except:
            messagebox.showerror(
                title="Error",
                message="Something went wrong, did you select an appropriately output CSV file?")
            return

        for i in range(1, l):
            self.label_text[i].delete(1.0, tk.END)
            self.label_text[i].insert(tk.END, self.label_name[i])
            self.regex_text[i].delete(1.0, tk.END)
            self.regex_text[i].insert(tk.END, self.phrases[i])

        self.enable_button()


    def enable_button(self):
        self.prev_button.config(state='normal')
        self.next_button.config(state='normal')
        self.add_ann_button.config(state='normal')
        self.del_ann_button.config(state='normal')
        self.save_button.config(state='normal')

        output_fname = self.regex_label.get()
        self.refresh_viewer(output_fname)

    def disable_button(self):
        self.prev_button.config(state='disabled')
        self.next_button.config(state='disabled')
        self.add_ann_button.config(state='disabled')
        self.del_ann_button.config(state='disabled')
        self.save_button.config(state='disabled')

    # Functions that change display
    def refresh_viewer(self, output_fname):
        def clean_phrase(phrase):
            cleaned = str(phrase.replace('||', '|').replace('\\r', '\\n'))
            cleaned = re.sub(r'(\n+|\r\r)', '\n', cleaned)
            cleaned = re.sub(r'( +|\t+)', ' ', cleaned)
            cleaned = re.sub(r'\r', '', cleaned)
            return str(cleaned.strip())

        def combine_keywords(text):
            df = pd.DataFrame(map(' '.join, zip(*[iter(text.split(' '))]*100)), columns=['text'])
            df['regex'] = df['text'].apply(
                        lambda x: 1 if any(re.search('(\W|^)' + p.lower(), x.lower()) for p in self.allphrases) else 0)
            if all(df['regex'] == 0):
                messagebox.showerror(title="Warning", message="No keywords found!")
                return
            return df[df['regex'] == 1].reset_index(drop=True)['text'].str.cat(sep='\n----\n')
        # run regex
        try:
            if not self.load_annotation:
                f = self.data_model.input_fname.split('.')[-1]
                if 'xls' in f:
                    self.data_model.input_df = pd.read_excel(
                        self.data_model.input_fname)
                    self.data_model.output_df = pd.read_excel(
                        self.data_model.input_fname, usecols=[
                            self.patient_key, self.note_key])
                else:
                    self.data_model.input_df = pd.read_csv(
                        self.data_model.input_fname)
                    self.data_model.output_df = pd.read_csv(
                        self.data_model.input_fname, usecols=[
                            self.patient_key, self.note_key])
                self.data_model.output_df[self.note_key] = self.data_model.output_df[self.note_key].astype(
                    str).apply(lambda x: clean_phrase(x))  
            else:
                self.data_model.input_df = self.data_model.output_df = pd.read_csv(
                    self.data_model.input_fname)

            if self.patientlevel: 
                # concate report text on patient's level
                self.data_model.output_df = self.data_model.output_df.groupby(self.patient_key)[self.note_key].apply(lambda x: '\n'.join(x)).to_frame().reset_index()
            # Display only positive hits
            if self.positivehit or self.patientlevel:
                self.allphrases = []
                for i in range(1, 4):
                    if self.phrases[i] != self.original_regex_text and len(
                            self.phrases[i]) > 0:
                        self.allphrases.extend(self.phrases[i].replace(', ',',').split(','))
                        if not self.load_annotation:
                            self.data_model.output_df['L%d_' %
                                                      i + self.label_name[i]] = None
                            self.data_model.output_df['L%d_' %
                                                      i + self.label_name[i] + '_span'] = None
                            self.data_model.output_df['L%d_' %
                                                      i + self.label_name[i] + '_text'] = None
                            self.data_model.output_df['K%d_' %
                                                      i + str(self.phrases[i])] = ''
                if self.patientlevel:
                    self.data_model.output_df[self.note_key] = self.data_model.output_df[self.note_key].apply(lambda x: combine_keywords(x)) 
                    self.data_model.nokeyword_df = []
                else: 
                    self.data_model.output_df['regex'] = self.data_model.output_df[self.note_key].apply(
                        lambda x: 1 if any(re.search('(\W|^)' + p.lower(), x.lower()) for p in self.allphrases) else 0)
                    if all(self.data_model.output_df['regex'] == 0):
                        self.data_model.output_df['regex'] = 1
                        messagebox.showerror(title="Warning",
                                            message="No keywords found!")
                    self.data_model.nokeyword_df = self.data_model.output_df[self.data_model.output_df['regex'] == 0].reset_index(drop=True)
                    self.data_model.output_df = self.data_model.output_df[self.data_model.output_df['regex'] == 1].reset_index(drop=True)
                    self.data_model.output_df = self.data_model.output_df.drop(columns=['regex'])
                    self.data_model.nokeyword_df = self.data_model.nokeyword_df.drop(columns=['regex'])
            else:
                self.data_model.nokeyword_df = []
        except BaseException:
            messagebox.showerror(
                title="Error",
                message="Something went wrong, did you select an appropriately columns?")
            return
        self.data_model.output_fname = output_fname
        self.refresh_model()


    def refresh_model(self):
        if not self.load_annotation:
            self.data_model.current_row_index = 0
        else:
            try:
                self.data_model.current_row_index = self.data_model.output_df.index[
                    self.data_model.output_df['L1_' + self.label_name[1]].isna()].tolist()[0]
            except BaseException:
                self.data_model.current_row_index = 0

        if self.data_model.input_fname:
            try:
                self.data_model.display_df = self.data_model.output_df.copy()
                self.data_model.num_notes = self.data_model.display_df.shape[0]
                self.display_output_note()
            except BaseException:
                pass


    def display_output_note(self):
        current_note_row = self.data_model.display_df.iloc[self.data_model.current_row_index]
        
        try:
            current_note_text = current_note_row[self.note_key]
        except BaseException:
            messagebox.showerror(
                title='Error',
                message='Unable to retrieve note text. Did you select the correct key?')
            return
        try:
            current_patient_id = current_note_row[self.patient_key]
        except BaseException:
            messagebox.showerror(
                title='Error',
                message='Unable to retrieve note ID. Did you select the correct key?')
            return
        
        self.number_label.config(
            text='%d of %d' %
            (self.data_model.current_row_index + 1, self.data_model.num_notes))
        if self.patientlevel: idtext = 'Patient ID: %s' % current_patient_id
        else: idtext = 'Note ID: %s' % current_patient_id
        self.patient_num_label.config(
            text=idtext)
        
        self.pttext.config(state=tk.NORMAL)
        self.pttext.delete(1.0, tk.END)
        self.pttext.insert(tk.END, current_note_text)
        self.pttext.config(state=tk.DISABLED)
        
        input_df = self.data_model.display_df.iloc[[
            self.data_model.current_row_index]]
        
        for i in range(1, 4):
            if self.phrases[i] != self.original_regex_text and len(
                    self.phrases[i]) > 0:
                match_indices = self.find_matches(
                    self.phrases[i],
                    "keyword_%d" %
                    i,
                    "L%d_" %
                    i +
                    self.label_name[i] +
                    '_span',
                    input_df)
                value = current_note_row["L%d_" %i + self.label_name[i]]
                if (not value or value.astype(str) == 'nan')and match_indices: value = '1'
                elif not value or value.astype(str) == 'nan' : value = '0'
                else: value = value.astype(str)
                self.ann_text[i].delete(0, tk.END)
                self.ann_text[i].insert(0, value)
        self.pttext.tag_raise("sel")
        self.length, l = {}, 0
        for i in range(1, int(self.pttext.index("end").split('.')[0])):
            self.length[i] = l
            l += int(self.pttext.index(str(i) + ".end").split('.')[1]) + 1


    def find_matches(
            self,
            phrases,
            keyword,
            label_name,
            input_df):
        match_indices = self.data_model.output_df.at[self.data_model.current_row_index, label_name]

        if match_indices and isinstance(match_indices, str):
            match_indices = [i.split(',') for i in match_indices.split('|')]
        else:
            match_indices = run_regex(
                input_df,
                phrases,
                self.data_model.current_row_index,
                False,
                self.note_key,
                self.patient_key)

        tag_start = '1.0'
        # Add highlighting
        for start, end in match_indices:
            pos_start = '{}+{}c'.format(tag_start, start)
            pos_end = '{}+{}c'.format(tag_start, end)
            self.pttext.tag_add(keyword, pos_start, pos_end)

        return match_indices


    def save_matches(self, keyword, label_name, value='1'):
        tags = self.pttext.tag_ranges(keyword)
        match = ''
        text = ''
        for i in range(0, len(tags), 2):
            s = str(tags[i]).split('.')
            e = str(tags[i + 1]).split('.')
            start = int(s[1]) + self.length[int(s[0])]
            end = int(e[1]) + self.length[int(e[0])]
            if i > 0:
                text += '|'
                match += '|'
            text += '{}'.format(self.pttext.get(tags[i], tags[i + 1]))
            match += '{},{}'.format(start, end)
        current_row_index = self.data_model.display_df.index[self.data_model.current_row_index]
        if match and value == '0': value = '1'
        self.data_model.output_df.at[current_row_index, label_name] = value
        self.data_model.output_df.at[current_row_index,
                                     label_name + '_span'] = match
        self.data_model.output_df.at[current_row_index,
                                     label_name + '_text'] = text


    def on_save_annotation(self):
        if self.data_model.output_fname[-4:] not in ['.csv', '.dta']:
            messagebox.showerror(
                title='Error',
                message='Did you key in the correct CSV or DTA output filename?')
        for i in range(1, 4):
            if self.phrases[i] != self.original_regex_text and len(
                    self.phrases[i]) > 0:
                value = self.ann_text[i].get()
                self.save_matches(
                    "keyword_%d" %
                    i,
                    'L%d_' %
                    i +
                    self.label_name[i],
                    value)
        if len(self.data_model.nokeyword_df) > 0:
            self.data_model.save_df = pd.concat(
                [self.data_model.output_df, self.data_model.nokeyword_df], axis=0, sort=False)
        else:
            self.data_model.save_df = self.data_model.output_df
        if not self.patientlevel and not self.load_annotation:
            self.data_model.save_df = pd.merge(
                self.data_model.save_df,
                self.data_model.input_df,
                on=self.patient_key)
        self.data_model.write_to_annotation()


    def on_prev(self):
        self.on_save_annotation()
        if self.data_model.current_row_index > 0:
            self.data_model.current_row_index -= 1
        self.display_output_note()

    def on_next(self):
        self.on_save_annotation()
        if self.data_model.current_row_index < self.data_model.num_notes:
            self.data_model.current_row_index += 1
        self.display_output_note()

    def on_add_annotation(self):
        self.modify_annotation('add')

    def on_delete_annotation(self):
        self.modify_annotation('delete')

    def modify_annotation(self, action):
        if self.pttext.tag_ranges(tk.SEL):
            if self.label == 1:
                keyword = "keyword_1"
            elif self.label == 2:
                keyword = "keyword_2"
            else:
                keyword = "keyword_3"
            s0 = self.pttext.index("sel.first").split('.')
            s1 = self.pttext.index("sel.last").split('.')
            pos_start = '{}.{}'.format(*s0)
            pos_end = '{}.{}'.format(*s1)
            self.pttext.tag_remove(tk.SEL, "1.0", tk.END)
            if action == 'add':
                self.pttext.tag_add(keyword, pos_start, pos_end)
            else:
                self.pttext.tag_remove(keyword, pos_start, pos_end)
        else:
            messagebox.showerror(
                title='Error',
                message='No text selected!')

    def clear_textbox(self, event, widget, original_text):
        if widget.get(1.0, 'end-1c') == original_text:
            widget.delete(1.0, 'end-1c')

    def on_positive_checkbox_click(self, event, widget):
        if self.positivehit:
            self.positivehit = False
        else:
            self.positivehit = True
        if self.data_model.output_df is not None:
            self.on_run_regex()
        self.refresh_model()

    def on_patient_checkbox_click(self, event, widget):
        if self.patientlevel:
            self.patientlevel = False
            self.patient_id_label.config(text='Note ID column: ')
        else:
            self.patientlevel = True
            self.patient_id_label.config(text='Patient ID column: ')

    def on_radio_click(self):
        self.label = self.radio_value.get()
        for i in range(1, 4):
            self.regex_text[i].configure(bg='white')
            self.label_text[i].configure(
                font=font.Font(family='Roboto', size=14))

        color = {1: '#ffe6ff', 2: '#e6e6ff', 3: '#fff2e6'}
        i = self.label
        self.regex_text[i].configure(bg=color[i])
        self.label_text[i].configure(
            font=font.Font(
                family='Roboto',
                size=14,
                weight='bold'))

    def setup_interface(self, root):
        # Define fonts
        titlefont = font.Font(family='Open Sans', size=18, weight='bold')
        boldfont = font.Font(size=16, family='Open Sans', weight='bold')
        textfont = font.Font(family='Roboto', size=14)
        labelfont = font.Font(family='Roboto', size=12)

        left_bg_color = 'lightblue1'
        right_bg_color = 'azure'
        # Creating all main containers
        left_frame = tk.Frame(root, bg=left_bg_color)
        right_frame = tk.Frame(root, bg=right_bg_color)

        # Laying out all main containers
        root.grid_rowconfigure(0, weight=1)
        for i in range(3):
            root.grid_columnconfigure(i, weight=1)

        left_frame.grid(
            column=0,
            row=0,
            columnspan=2,
            rowspan=2,
            sticky='nsew')
        for i in range(11):
            left_frame.grid_rowconfigure(i, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_columnconfigure(1, weight=1)

        right_frame.grid(column=2, row=0, columnspan=2, sticky='nsew')
        for i in range(11):
            right_frame.grid_rowconfigure(i, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        ########################################################
        #                      Left Frame                      #
        ########################################################
        # Left header first line
        header_frame = tk.Frame(left_frame, bg=left_bg_color)
        header_frame.grid(
            column=0,
            row=0,
            columnspan=2,
            padx=10,
            pady=10,
            sticky='nsew')
        header_frame.grid_propagate(False)
        header_frame.grid_rowconfigure(0, weight=1)
        header_frame.grid_rowconfigure(1, weight=1)
        header_frame.grid_columnconfigure(0, weight=2)
        header_frame.grid_columnconfigure(1, weight=1)
        header_frame.grid_columnconfigure(2, weight=1)

        title_text = tk.Label(
            header_frame,
            text='Clinical Note',
            font=titlefont,
            bg=left_bg_color)
        title_text.grid(column=0, row=0, sticky='w')

        # Left header second line
        button_frame = tk.Frame(header_frame, bg=left_bg_color)
        button_frame.grid(column=0, row=1, columnspan=1, sticky='nsew')
        button_frame.grid_propagate(False)
        for i in range(3):
            button_frame.grid_columnconfigure(i, weight=1)
        button_frame.grid_rowconfigure(0, weight=1)
        button_frame.grid_rowconfigure(1, weight=1)

        self.prev_button = tk.Button(
            button_frame,
            text='Prev',
            width=5,
            state='disabled',
            command=self.on_prev)
        self.prev_button.grid(column=0, row=0, sticky='sw')

        self.number_label = tk.Label(
            button_frame,
            font=labelfont,
            text='',
            bg=left_bg_color)
        self.number_label.grid(column=1, row=0, sticky='sw')

        self.next_button = tk.Button(
            button_frame,
            text='Next',
            width=5,
            state='disabled',
            command=self.on_next)
        self.next_button.grid(column=2, row=0, sticky='sw')

        # Patient ID
        self.patient_num_label = tk.Label(
            header_frame, text='', font=labelfont, bg=left_bg_color)
        self.patient_num_label.grid(column=1, row=1)

        # Filter checkbox
        positive_checkbox_var = tk.BooleanVar()
        self.positive_checkbox = tk.Checkbutton(
            header_frame,
            text='Display only positive hits',
            font=labelfont,
            variable=positive_checkbox_var,
            bg=left_bg_color,
            offvalue=False,
            onvalue=True)
        self.positive_checkbox.var = positive_checkbox_var
        self.positive_checkbox.grid(column=2, row=1, sticky='e')
        self.positive_checkbox.bind(
            "<Button-1>",
            lambda event: self.on_positive_checkbox_click(
                event,
                self.positive_checkbox))
        chkValue = tk.BooleanVar()
        chkValue.set(False)
        self.positive_checkbox.config(var=chkValue)

        # Text frame
        text_frame = tk.Frame(left_frame, borderwidth=1, relief="sunken")
        text_frame.grid(
            column=0,
            row=1,
            rowspan=9,
            columnspan=2,
            padx=10,
            pady=0,
            sticky='nsew')
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_propagate(False)

        # Modify annotation
        self.add_ann_button = tk.Button(
            left_frame,
            text='Add',
            font=textfont,
            width=15,
            state='disabled',
            command=self.on_add_annotation)
        self.add_ann_button.grid(column=0, row=10, padx=10, sticky='we')

        self.del_ann_button = tk.Button(
            left_frame,
            text='Delete',
            font=textfont,
            width=15,
            state='disabled',
            command=self.on_delete_annotation)
        self.del_ann_button.grid(column=1, row=10, padx=10, sticky='we')

        # Patient note container (with scrolling)
        self.pttext = tk.Text(
            text_frame,
            wrap="word",
            font=textfont,
            background="white",
            borderwidth=0,
            highlightthickness=0)
        scrollbar = tk.Scrollbar(text_frame)
        self.pttext.config(yscrollcommand=scrollbar.set)
        self.pttext.config(state=tk.DISABLED)
        scrollbar.config(command=self.pttext.yview)
        scrollbar.grid(column=1, row=0, sticky='nsw')
        self.pttext.grid(column=0, row=0, padx=15, pady=15, sticky='nsew')
        self.pttext.tag_config('keyword_1', background='#ffd2ff')
        self.pttext.tag_config('keyword_2', background='#ccccff')
        self.pttext.tag_config('keyword_3', background='#ffe6cc')
        self.pttext.bind("<1>", lambda event: self.pttext.focus_set())

        ########################################################
        #                      Right Frame                     #
        ########################################################
        # Right upper frame
        right_upper_frame = tk.Frame(right_frame, bg=right_bg_color)
        right_upper_frame.grid(
            column=0,
            row=0,
            padx=10,
            pady=10,
            sticky='nsew')
        right_upper_frame.grid_propagate(False)
        right_upper_frame.grid_rowconfigure(0, weight=1)
        right_upper_frame.grid_columnconfigure(0, weight=1)
        right_upper_frame.grid_columnconfigure(1, weight=1)

        in_file_text = tk.Label(
            right_upper_frame,
            text='Input File',
            font=labelfont,
            bg=right_bg_color)
        in_file_text.grid(column=0, row=0, sticky='nsw')

        file_button = tk.Button(
            right_upper_frame,
            text='Select',
            command=self.on_select_file,
            bg=right_bg_color)
        file_button.grid(column=1, row=0, sticky='e')

        self.file_text = tk.Label(
            right_upper_frame,
            text='',
            bg=right_bg_color,
            font=labelfont,
            fg='dodgerblue4')
        self.file_text.grid(column=1, row=0, sticky='nsw')

        out_file_text = tk.Label(
            right_upper_frame,
            text='Output File',
            font=labelfont,
            bg=right_bg_color)
        out_file_text.grid(column=0, row=1, sticky='nsw')

        self.regex_label = tk.Entry(right_upper_frame, font=labelfont)
        self.regex_label.insert(0, 'output.csv')
        self.regex_label.grid(column=1, row=1, sticky='nswe')

        # Right upper regex options container
        self.right_options_frame = tk.Frame(right_frame, bg=right_bg_color)
        self.right_options_frame.grid(
            column=0, row=1, rowspan=1, padx=10, sticky='nsew')
        self.right_options_frame.grid_propagate(False)
        self.right_options_frame.grid_columnconfigure(0, weight=1)
        self.right_options_frame.grid_columnconfigure(1, weight=1)
        for i in range(4):
            self.right_options_frame.grid_rowconfigure(i, weight=1)

        patient_checkbox_var = tk.BooleanVar()
        self.patient_checkbox = tk.Checkbutton(
            self.right_options_frame,
            text='On patient level',
            font=labelfont,
            bg=right_bg_color,
            variable=patient_checkbox_var,
            offvalue=False,
            onvalue=True)
        self.patient_checkbox.var = patient_checkbox_var
        self.patient_checkbox.grid(column=0, row=1, columnspan=2, sticky='ns')
        self.patient_checkbox.bind(
            "<Button-1>",
            lambda event: self.on_patient_checkbox_click(
                event,
                self.patient_checkbox))
        chkValue = tk.BooleanVar()
        chkValue.set(False)
        self.patient_checkbox.config(var=chkValue)

        self.patient_id_label = tk.Label(
            self.right_options_frame,
            text='Note ID column: ',
            font=labelfont,
            bg=right_bg_color)
        self.patient_id_label.grid(column=0, row=2, sticky='nsw')

        self.note_key_entry_label = tk.Label(
            self.right_options_frame,
            text='Report text column: ',
            font=labelfont,
            bg=right_bg_color)
        self.note_key_entry_label.grid(column=0, row=3, sticky='nsw')

        # Right label container
        # Label 1
        right_label_frame = tk.Frame(right_frame, bg=right_bg_color)
        for i in range(13):
            if i in [2,6,10]: right_label_frame.grid_rowconfigure(i, weight=6)
            else: right_label_frame.grid_rowconfigure(i, weight=1)
        right_label_frame.grid_columnconfigure(0, weight=1)
        right_label_frame.grid_columnconfigure(1, weight=3)
        right_label_frame.grid(
            column=0,
            row=2,
            rowspan=7,
            padx=10,
            pady=10,
            sticky='nsew')
        right_label_frame.grid_propagate(False)

        regex_title = tk.Label(
            right_label_frame,
            text='Labels',
            font=boldfont,
            bg=right_bg_color)
        regex_title.grid(column=0, row=0, columnspan=2, sticky='nswe')

        self.label_name = {}
        self.label_text = {}
        self.regex_text = {}
        self.ann_text = {}
        self.radio, self.ann = {}, {}
        self.original_label_text = {}
        self.original_regex_text = "Type comma-separated regex/keywords here."

        self.radio_value = tk.IntVar()
        self.radio_value.set(1)
        self.label = 1
        label_color = {1: '#ff99ff', 2: '#cc99ff', 3: '#ffcc99'}
        text_color = {1: '#ffe6ff', 2: 'white', 3: 'white'}
        rowstart = 0

        # Label 1
        for i in range(1,4):
            self.radio[i] = tk.Radiobutton(
                right_label_frame,
                text='',
                variable=self.radio_value,
                value=i,
                font=textfont,
                bg=label_color[i],
                command=self.on_radio_click)
            self.radio[i].grid(column=0, row=rowstart+1, sticky='nsew')

            self.original_label_text[i] = "Label_%d"%i
            self.label_text[i] = tk.Text(
                right_label_frame,
                font=font.Font(family='Roboto', size=14, weight='bold'),
                highlightthickness=0,
                height=1,
                width=30,
                bg=label_color[i])
            self.label_text[i].insert(tk.END, self.original_label_text[i])
            self.label_text[i].grid(column=1, row=rowstart+1, sticky='nswe')

            self.regex_text[i] = tk.Text(
                right_label_frame,
                font=labelfont,
                borderwidth=1,
                highlightthickness=0,
                height=2,
                bg=text_color[i])
            self.regex_text[i].insert(tk.END, self.original_regex_text)
            self.regex_text[i].grid(column=0, row=rowstart+2, columnspan=2, sticky='nsew')

            self.ann[i] = tk.Label(
                right_label_frame,
                text='Value',
                font=labelfont,
                bg=right_bg_color)
            self.ann[i].grid(column=0, row=rowstart+3, sticky='nw')
            self.ann_text[i] = tk.Entry(right_label_frame, font=textfont)
            self.ann_text[i].grid(column=1, row=rowstart+3, sticky='new')
            rowstart += 4

        self.label_text[1].bind(
            "<Button-1>",
            lambda event: self.clear_textbox(
                event,
                self.label_text[1],
                self.original_label_text[1]))
        self.regex_text[1].bind(
            "<Button-1>",
            lambda event: self.clear_textbox(
                event,
                self.regex_text[1],
                self.original_regex_text))
        self.label_text[2].bind(
            "<Button-1>",
            lambda event: self.clear_textbox(
                event,
                self.label_text[2],
                self.original_label_text[2]))
        self.regex_text[2].bind(
            "<Button-1>",
            lambda event: self.clear_textbox(
                event,
                self.regex_text[2],
                self.original_regex_text))
        self.label_text[3].bind(
            "<Button-1>",
            lambda event: self.clear_textbox(
                event,
                self.label_text[3],
                self.original_label_text[3]))
        self.regex_text[3].bind(
            "<Button-1>",
            lambda event: self.clear_textbox(
                event,
                self.regex_text[3],
                self.original_regex_text))


        # Right button container
        right_button_frame = tk.Frame(right_frame, bg=right_bg_color)
        right_button_frame.grid(
            column=0, row=9, rowspan=2, padx=10, sticky='nsew')
        right_button_frame.grid_propagate(False)
        right_button_frame.grid_columnconfigure(0, weight=1)
        for i in range(3):
            right_button_frame.grid_rowconfigure(i, weight=1)

        self.regex_button = tk.Button(
            right_button_frame,
            text='Run Regex',
            font=textfont,
            state='disabled',
            command=self.on_run_regex)
        self.regex_button.grid(column=0, row=0, sticky='nwe')

        self.load_button = tk.Button(
            right_button_frame,
            text='Load annotation',
            font=textfont,
            state='disabled',
            command=self.on_load_annotation)
        self.load_button.grid(column=0, row=1, sticky='nwe')
        self.load_annotation = False

        self.save_button = tk.Button(
            right_button_frame,
            text='Save',
            font=textfont,
            state='disabled',
            command=self.on_save_annotation)
        self.save_button.grid(column=0, row=2, sticky='nwe')
