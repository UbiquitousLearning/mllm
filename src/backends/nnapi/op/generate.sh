#!/bin/bash

# List of file names (without extensions)
file_names=(
    "NNAPIAdd"
    "NNAPILinear"
    "NNAPIMatmul"
    "NNAPISoftmax"
)

template_header="tamplate.hpp"
template_source_file="template.cpp"

# Iterate through the list of file names
for name in "${file_names[@]}"; do
    header_file="${name}.hpp"
    source_file="${name}.cpp"
    # convert name to UPPER CASE
    name_uppercase=$(echo "$name" | tr '[:lower:]' '[:upper:]')

    # 读取模板文件内容
    template_content=$(<"$template_file")
    class_name="MyClass"
    output_content="${template_content//\{\{CLASS_NAME\}\}/$class_name}"

    # Create the header file
    touch "$header_file"
    echo "#ifndef MLLM_${name_uppercase}_H" >> "$header_file"
    echo "#define MLLM_${name_uppercase}_H" >> "$header_file"
    echo "" >> "$header_file"
    echo "#include \"Backend.hpp\"" >> "$header_file"
    echo "#include \"Op.hpp\"" >> "$header_file"
    echo "#include \"Types.hpp\"" >> "$header_file"
    echo "" >> "$header_file"
    echo "namespace mllm {" >> "$header_file"
    echo "" >> "$header_file"
    # TODO: add class definition from template
    echo "} // namespace mllm" >> "$header_file"
    echo "#endif // MLLM_${name_uppercase}_H" >> "$header_file"

    # Create the source file
    touch "$source_file"
    echo "#include \"$header_file\"" >> "$source_file"
    echo "" >> "$source_file"
    echo "namespace mllm {" >> "$source_file"
    echo "" >> "$source_file"
    # TODO: add class implementation from template
    echo "} // namespace mllm" >> "$source_file"

    echo "Created $header_file and $source_file"
done
