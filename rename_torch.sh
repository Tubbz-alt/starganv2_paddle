find . -name "*.py" -exec sed -i -e 's/paddle_torch/paddorch/g' {} +
find . -name "*.py" -exec sed -i -e 's/torch/porch/g' {} + 
