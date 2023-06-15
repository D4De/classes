import enum

class OperatorType(enum.Enum):
    """
    The TensorFlow's injectables operators.
    This class stores the type (name of the operator) and allows
    to convert the operator type from TensorFlow's name to CAFFE model's name.
    """
    Conv2D1x1 = 1  # Convolution 2D with kernel size of 1.
    Conv2D3x3 = 2  # Convolution 2D with kernel size of 3.
    Conv2D3x3S2 = 3  # Convolution 2D with kernel size of 3 and stride of 2.
    AddV2 = 4  # Add between two tensors.
    BiasAdd = 5  # Add between a tensor and a vector.
    Mul = 6  # Mul between a tensor and a scalar.
    FusedBatchNormV3 = 7  # Batch normalization.
    RealDiv = 8  # Division between a tensor and a scalar.
    Exp = 9  # Exp activation function.
    LeakyRelu = 10  # Leaky Relu activation function.
    Sigmoid = 11  # Sigmoid activation function.
    Add = 12  # Add between two tensors.
    Conv2D = 13
    FusedBatchNorm = 14

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_model_name(self):
        """
        Returns the CAFFE model's name from the TensorFlow's operator type.

        Returns:
            [str] -- The CAFFE model's name.
        """
        # Switch statement that map each TF's operator type to the model names
        # used in the simulations.
        if self == OperatorType.Conv2D1x1:
            return "S1_conv"
        elif self == OperatorType.Conv2D3x3:
            return "S1_conv"
        elif self == OperatorType.Conv2D3x3S2:
            return "S3_convolution"
        elif self == OperatorType.AddV2:
            return "S1_add"
        elif self == OperatorType.BiasAdd:
            return "S1_biasadd"
        elif self == OperatorType.Mul:
            return "S1_mul"
        elif self == OperatorType.FusedBatchNormV3:
            return "S1_batchnorm"
        elif self == OperatorType.Exp:
            return "S1_exp"
        elif self == OperatorType.LeakyRelu:
            return "S1_leaky_relu"
        elif self == OperatorType.Sigmoid:
            return "S1_sigmoid"
        elif self == OperatorType.RealDiv:
            return "S1_div"
        elif self == OperatorType.Add:
            return "S2_add"
        elif self == OperatorType.Conv2D:
            return "S1_conv"
        elif self == OperatorType.FusedBatchNorm:
            return "S1_batchnorm"
        else:
            raise ValueError("Unable to find a model for this operator: {}".format(self))

    @staticmethod
    def all():
        """
        Returns all the types as list.

        Returns:
            [list] -- List of operator types.
        """
        return list(OperatorType)

    @staticmethod
    def all_aliases():
        """
        Returns the model's names associated to each operator type.

        Returns:
            [list] -- List of model's names
        """
        return [operator.get_model_name() for operator in OperatorType.all()]

