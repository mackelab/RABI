import glob, os
from pathlib import Path
# creating a variable and storing the text
# that we want to search
search_text = "sans-serif"

# creating a variable and storing the text
# that we want to add
replace_text = "Arial"
folder = str(os.path.dirname(__file__))
for file in Path(folder).rglob("*.svg"):
    file = str(file)
    print(file)


    # Opening our text file in read only
    # mode using the open() function
    with open(file, 'r') as f:

        # Reading the content of the file
        # using the read() function and storing
        # them in a new variable
        data = f.read()

        # Searching and replacing the text
        # using the replace() function
        data = data.replace(search_text, replace_text)

    # Opening our text file in write only
    # mode to write the replaced content
    with open(file, 'w') as f:

        # Writing the replaced data in our
        # text file
        f.write(data)

    # Printing Text replaced
    print("Text replaced")
