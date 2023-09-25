CLASS_NAME::CLASS_NAME(Backend *bn) : Op(bn) {
}

ErrorCode CLASS_NAME::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CLASS_NAME  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CLASS_NAME::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CLASS_NAME  setUp" << std::endl;
    return NO_ERROR;
}

ErrorCode CLASS_NAME::load(ParamLoader &loader) {
    std::cout << "CLASS_NAME load" << std::endl;
    return NO_ERROR;
}