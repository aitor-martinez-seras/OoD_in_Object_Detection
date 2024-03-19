# Convert lauch.json to args
# ------------------------------------------------- #
args_list = []
# ------------------------------------------------- #
# Transform the list of strings into a single string
only_string = " ".join(args_list)
print(only_string)

# ************************************************************************* #

# Convert args to launch json
# Transform the string into a list of strings and print with double quotes
# ------------------------------------------------- #
string_with_args = "python custom_training.py -e 300 --lr 0.001 --lrf 0.15 --owod_task t1 --model yolov5 --model_size l --devices 3 --dataset owod --batch_size 16 -cl_ms 5 --workers 12 --val_every 5 --from_scratch"
# ------------------------------------------------- #
args_list = string_with_args.split(" ")
# Now print the list with double quotes and commas between each element
print('"' + '", "'.join(args_list) + '"')