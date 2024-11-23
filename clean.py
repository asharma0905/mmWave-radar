def remove_string_from_file(file_path, string_to_remove):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Remove occurrences of the string
    modified_content = content.replace(string_to_remove, '')

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(modified_content)

def remove_paragraph_breaks(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove paragraph breaks ('\n')
    modified_lines = []
    for line in lines:
        if line.strip() != '':
            modified_lines.append(line)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(''.join(modified_lines))

# Example usage
file_path = 'example.txt'  # Replace 'example.txt' with your file path
string_to_remove = 'elapsed_time,frame_number,nPoint,point_x,point_y,point_z,velocity,intensity'  # Replace example string with the string you want to remove
remove_string_from_file(file_path, string_to_remove)
remove_paragraph_breaks(file_path)