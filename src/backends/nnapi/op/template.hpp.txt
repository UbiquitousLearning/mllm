class CLASS_NAME : public Op {
public:
    CLASS_NAME(Backend *bn);
    virtual ~CLASS_NAME() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

    virtual ErrorCode load(ParamLoader &loader) override;

private:
    bool support_multi_thread_ = false;
};

class CLASS_NAMECreator : public NNAPIBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn) const {
        return new CLASS_NAME(bn, false);
    }
};