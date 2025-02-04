import re

# Read the original requirements file
with open("requirements.txt", "r") as file:
    lines = file.readlines()

# Process each line
corrected_lines = []
for line in lines:
    # Remove the `=pypi_0` suffix
    line = re.sub(r"=pypi_0", "", line)
    
    # Replace `=` with `==` only when specifying versions
    line = re.sub(r"(?<=\w)=(?=\d)", "==", line)
    
    corrected_lines.append(line)

# Write back the corrected lines
with open("requirements_fixed.txt", "w") as file:
    file.writelines(corrected_lines)

print("Fixed requirements file saved as requirements_fixed.txt")
