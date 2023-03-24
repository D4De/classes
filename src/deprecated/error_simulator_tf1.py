import numpy as np
import tensorflow as tf
import time
import enum
from collections import OrderedDict, defaultdict
import json
from injection_sites_generator import *
import logging


class CampaignStatistics(object):
    def __init__(self):
        self.__stats = {}
        for outcome in Outcome:
            self.__stats[outcome] = 0

    def update(self, outcomes):
        try:
            for outcome in outcomes:
                self.__stats[outcome] += 1
        except:
            self.__stats[outcomes] += 1

    def reduce(self):
        reduced_stats = {}
        total_count = 0
        total_measurable_count = 0
        for outcome in self.__stats:
            if outcome != Outcome.RAW_VALUE:
                total_measurable_count += self.__stats[outcome]
            total_count += self.__stats[outcome]
        for outcome in self.__stats:
            if outcome != Outcome.RAW_VALUE:
                reduced_stats[outcome] = float(self.__stats[outcome]) / float(total_measurable_count)
            else:
                reduced_stats[outcome] = float(self.__stats[outcome]) / float(total_count)
        return reduced_stats

    def __str__(self):
        reduced_stats = self.reduce()
        fields = []
        for outcome in reduced_stats:
            fields.append("{} - {:.6f} - {}".format(outcome.name, reduced_stats[outcome], self.__stats[outcome]))
        return "\n".join(fields)


class Outcome(enum.IntEnum):
    MASKED = 1
    SDC = 2
    CRASHED = 3
    RAW_VALUE = 4

    def __init__(self, value):
        self.set_values(0, "", [])

    def __str__(self):
        return "{} - Run count: {} - Operator: {}".format(self.name, self.run_count, self.operator_name)

    def __hash__(self):
        return hash(self.value)

    def set_values(self, run_count, operator_name, raw_value):
        self.__run_count = run_count
        self.__operator_name = operator_name
        self.__raw_value = raw_value
        return self

    run_count = property(lambda self: self.__run_count)
    operator_name = property(lambda self: self.__operator_name)
    raw_value = property(lambda self: self.__raw_value)


class FaultInjector(object):
    """
    TensorFlow's error injector.
    Given a TensorFlow's session, it replicates the graph, together with variables' data,
    to instrument the model, extracts the operators activated during the inference phase,
    generates the injection sites and inserts them into the model.
    """

    def get_injection_site(self, idx):
        inj_sites = {idx: self.__injection_sites[0].to_json()}
        return inj_sites

    def save_injection_sites(self, path, mode="w"):
        result = {}
        last_key = 0
        if mode == "a":
            with open(path) as f:
                result = json.load(f)
            keys = [int(x) for x in result.keys()]
            last_key = sorted(keys)[-1]

        for index, inj_site in enumerate(self.__injection_sites):
            result[str(index + last_key + 1)] = inj_site.to_json()

        with open(path, "w") as f:
            json.dump(result, f)

    def __init__(self, session):
        """
        Initialize the error injector with an active TensorFlow's session.

        Arguments:
            session {tf.Session} -- TensorFlow's session containing the model.
        """
        # Stores the session and the graph contained.
        self.__old_session = session
        self.__old_graph = session.graph
        # The list of activated operators during the inference phase.
        self.__activated_operations = []
        # The session's configuration to be used in the fault session.
        # It minimizes the memory utilization,
        self.__memory_configuration = tf.ConfigProto()
        self.__memory_configuration.gpu_options.allow_growth = True
        self.__memory_configuration.log_device_placement = False
        # Dictionary which contains the map among the operators of the original graph
        # and the ones replicated in the fault graph.
        self.__fault_map = {}
        # Replicated graph in which errors are inserted.
        self.__fault_graph = None
        # Replicated session, which will run the fault graph, with the memory configuration mentioned above.
        self.__fault_session = None
        # Replicated fetches, which are used to map the fetches with the fault ones.
        self.__fault_fetches = None
        # Flag that indicates if the instrumentation phase has been done or not.
        self.__instrumentation_done = False
        # Histogram that contains the operator types as keys and the operator names as values.
        self.__operations_histogram = {}
        # Structure that holds an elaborated version of the operation histogram.
        # The keys are the operator types and the values are lists of couple (operator name, output size).
        self.__injectable_sites = {}
        # Operator types that can be injected, according the models extracted.
        self.__injectable_operators = [
            "Add",
            "AddV2",
            "FusedBatchNormV3",
            "FusedBatchNorm",
            "BiasAdd",
            "Conv2D",
            "RealDiv",
            "Exp",
            "LeakyRelu",
            "Mul",
            "Sigmoid",
            "Conv2D1x1",
            "Conv2D3x3",
            "Conv2D3x3S2",
            "Conv2D_test",
            "FusedBatchNorm_test"
        ]

    def instrument(self, fetches, feeds):
        """
        Performs the instrumentation phase needed to be able to inject errors.
        During this phase, two graphs are built. The first one is used to extract
        the activated operators during the inference phase, given the fetches and feeds.
        The latter will be the graph in which errors are injected.
        The two graphs ensure that the original session and graph will remain untouched.
        Fetches and feeds are needed because the instrumentation phase needs a real inference case,
        so fetches and feeds must be the same as in the injection phase.
        Arguments:

            fetches {list(tf.Tensor)} -- TensorFlow's model fetches.

            feeds {dict{tf.Tensor: object}} -- TensorFlow's model feeds.
        """

        def activation(*inputs):
            """
            Passthrough function that takes the type, name and outputs of a TensorFlow's operator and
            updates the operators' histogram.
            """
            # Inputs[0] is the operator name.
            # Inputs[1] is the operator type.
            # Stores the operator name in the list of activated operators (legacy operation), no needed anymore.
            self.__activated_operations.append(inputs[0])
            # Append to the operator name to the list of operators for that type.
            if inputs[1] not in self.__operations_histogram:
                self.__operations_histogram[inputs[1]] = [inputs[0]]
            else:
                self.__operations_histogram[inputs[1]].append(inputs[0])
            return inputs

        start_time = time.time()  # Start time of the instrumentation phase.
        activations_graph = tf.Graph()  # Creates a new graph, which will be used to extract the activated operators.
        activations_fault_map = {}  # Creates a map among the original graph operators and the activation graph.
        # Activation graph and fault map are temporary, will be destroyed at the end of the instrumentation phase.
        with activations_graph.as_default():
            # Sets the activation graph as default so any insertion will be done in that graph.
            with activations_graph.name_scope(
                    "act"):  # Sets the name scope as "act" so each new node will have that prefix.
                # Slides through each operator of the original graph.
                for idx, operation in enumerate(self.__old_graph.get_operations()):
                    # print("Operazione numero: ", idx)
                    # Returns the control inputs of the operator according the activations fault map.
                    # Control inputs are nodes that need to be executed before the current one.
                    replicated_control_inputs = self.__get_replicated_inputs(operation.control_inputs,
                                                                             fault_map=activations_fault_map)
                    # Checks if the operator type is one of the injectable.
                    if str(operation.type) in self.__injectable_operators:
                        # Instantiates the passthrough function in the activations graph.
                        # The inputs are the name and type of the operator.
                        activation_operation = tf.py_func(
                            activation,
                            [tf.constant(operation.name, dtype=tf.string),
                             tf.constant(str(operation.type), dtype=tf.string)],
                            [tf.string, tf.string]
                        )[0].op
                        # Forces the passthrough function to be a dependency for the current operator,
                        # so it will be executed before the operator.
                        replicated_control_inputs.append(activation_operation)
                    # Enforces the dependency.
                    with activations_graph.control_dependencies(replicated_control_inputs):
                        # Gets the operator inputs according the activation fault map.
                        replicated_inputs = self.__get_replicated_inputs(operation.inputs,
                                                                         fault_map=activations_fault_map)
                        # Creates the operator replica.
                        replicated_operation = self.__create_operator_replica(
                            activations_graph,
                            operation,
                            replicated_inputs
                        )
                        # Adds pairs of original output and replicated output in the activations fault map.
                        self.__set_replicated_outputs(zip(operation.outputs, replicated_operation.outputs),
                                                      fault_map=activations_fault_map)
                        # Sets the replicated operation in the activations fault map.
                        activations_fault_map[operation.name] = replicated_operation
                # Creates a temporary TensorFlow's session with the memory configuration.
                with tf.Session(graph=activations_graph, config=self.__memory_configuration) as activations_session:
                    # Resets the list and the dictionary.
                    self.__activated_operations = []
                    self.__operations_histogram = {}
                    # Transfers the parameters from the original session to the temporary.
                    self.__assign_parameters(activations_session, "act/")
                    # Executes the model with the fetches and feeds.
                    self.__run(fetches, feed_dict=feeds, session=activations_session, fault_map=activations_fault_map)
        # After the run, is possible to extract the activated operators from the histogram and build a data-structure
        # that is usable by the InjectionSitesGenerator.
        self.__extract_injectable_sites(self.__old_graph, self.__operations_histogram)
        with open("injectable_sites.json", "w") as injectable_sites_json_file:
            json.dump(self.__injectable_sites, injectable_sites_json_file, sort_keys=True)
        self.__fault_graph = tf.Graph()  # Creates a new graph, which will used for the error injections.
        self.__fault_map = {}  # Creates a new fault map.
        self.__fault_fetches = {}  # Creates a new fetches fault map, among the original fetches and the replicated ones
        with self.__fault_graph.as_default():  # Sets the fault graph as default.
            with self.__fault_graph.name_scope("fi"):  # Sets a new prefix for the graph nodes.
                # Slides through each operator.
                for operation in self.__old_graph.get_operations():
                    # Gets the replicated control inputs.
                    replicated_control_inputs = self.__get_replicated_inputs(operation.control_inputs)
                    # Enforce the dependency constraint to be met.
                    with self.__fault_graph.control_dependencies(replicated_control_inputs):
                        # Gets the replicated inputs from the fault map.
                        replicated_inputs = self.__get_replicated_inputs(operation.inputs)
                        # Replicates the operator.
                        replicated_operation = self.__create_operator_replica(
                            self.__fault_graph,
                            operation,
                            replicated_inputs
                        )
                        replicated_outputs = replicated_operation.outputs
                        # Sets the outputs in the fault map and fetches.
                        self.__fault_fetches[operation.name] = replicated_outputs
                        self.__set_replicated_outputs(zip(operation.outputs, replicated_outputs))
                        self.__fault_map[operation.name] = replicated_operation
        # Creates the fault session and assigns the parameters from the original session to the fault one.
        self.__fault_session = tf.Session(graph=self.__fault_graph, config=self.__memory_configuration)
        self.__assign_parameters(self.__fault_session, "fi/")
        end_time = time.time()
        print("Instrumentation time: {:.5f} s".format(end_time - start_time))
        print("{} operations have been activated".format(sum([len(i) for i in self.__injectable_sites.values()])))
        # Sets that the instrumentation has been concluded.
        self.__instrumentation_done = True

    def generate_injection_sites(self, mode, size, operator_type=None, models_mode='', **kwargs):
        """
        Generates the injection sites, after the instrumentation phase, according to the mode given.\n
        Mode "RANDOM": creates the injection sites by considering all the operators.\n
        Mode "OPERATOR": creates the injection sites by considering only the operators of the chosen type.\n
        Mode "OPERATOR_SPECIFIC": creates the injection sites by considering only one chosen specific operator.\n
        The injection sites will be used later in the injection phase.

        Arguments:

            mode {str} -- String that contains the mode at which the injection sites will be generated.

            size {int} -- The numbers of injection sites to be generated.

        Raises:
            ValueError: raises an exception when the mode is not recognized.
        """

        def get_user_choice(options):
            """
            Asks the user to choose among one of the given options.

            Arguments:

                options {list} -- List of printable options.

            Raises:
                ValueError: raises an exception if the option chosen is outside the boundaries.

            Returns:
                [object] -- Returns the option selected.
            """
            # Prints each option with a index associated.
            for index, option in enumerate(options):
                print("Option {}: {}".format(index + 1, option))
            # Gets the index chosen.
            # No error validation, just crash.
            option_chosen = int(raw_input("Answer: "))
            # Checks only if the chosen option lies in boundaries.
            if option_chosen <= 0 or option_chosen > len(options):
                raise ValueError("Invalid option chosen")
            # Returns the option chosen.
            # The index is 1-based.
            return options[option_chosen - 1]

        def to_injectable_site(dic):
            """
            Given the injectable sites dictionary, with operator types as keys and couple of name and output size as values.
            Returns a list of InjectableSite that are used by the InjectionSitesGenerator.

            Arguments:

                dic {dict} -- Injectables sites.

            Returns:

                [list(InjectableSite)] -- Returns a list of converted injectable sites.
            """
            injectable_sites_class = []
            for key in dic:
                for elem in dic[key]:
                    # Convert each injectble site from dictionary to the class.
                    try:
                        injectable_sites_class.append(InjectableSite(OperatorType[key], elem["name"], elem["size"]))
                    except KeyError:
                        pass
            return injectable_sites_class

        if mode == "RANDOM":
            # Instanties the InjectionSitesGenerator and generates a random injection with the injectable sites and
            # the campaign size.
            self.__injection_sites = InjectionSitesGenerator(
                to_injectable_site(self.__injectable_sites)).generate_random_injection_sites(size)
        elif mode == "OPERATOR":
            # Asks the user the type of operator he wants to inject.
            if operator_type is None:
                operator_type = get_user_choice(self.__injectable_sites.keys())
            # Creates a copy of the injectables sites.
            injectable_sites_copy = self.__injectable_sites.copy()
            # Pops all the keys except the selected type.
            for key in self.__injectable_sites.keys():
                if key != operator_type:
                    injectable_sites_copy.pop(key)
            # Instanties the InjectionSitesGenerator and generates a random injection with the injectable sites and
            # the campaign size.
            try:
                assert len(injectable_sites_copy) > 0
            except AssertionError:
                return False
            self.__injection_sites = InjectionSitesGenerator(
                to_injectable_site(injectable_sites_copy)).generate_random_injection_sites(size)

            if len(self.__injection_sites) == 0:
                return False

            return True
        elif mode == "OPERATOR_SPECIFIC":
            # Asks the user the type of operator he wants to inject.
            # if "op" in kwargs:
            #     instance = (kwargs["op"], kwargs["op"])
            #     found = False
            #     for op_type in self.__injectable_sites.keys():
            #         for injectable_site in self.__injectable_sites[op_type]:
            #             if injectable_site["name"] == kwargs["op"]:
            #                 operator_type = op_type
            #                 found = True
            #                 break
            #         if found:
            #             break
            if "op_instance" in kwargs:
                instance = kwargs["op_instance"]

            else:
                operator_type = get_user_choice(self.__injectable_sites.keys())
                # Prepares the instances to be presented as operator name and output size.
                instances = [(i["name"], i["size"]) for i in self.__injectable_sites[operator_type]]
                # Gets the operator name, the user wants to inject.
                instance = get_user_choice(instances)
            injectable_sites_copy = self.__injectable_sites.copy()
            injection_site = None
            # Creates a single injection site only for the selected operator.
            for key in self.__injectable_sites.keys():
                if key != operator_type:
                    injectable_sites_copy.pop(key)
                else:
                    for elem in injectable_sites_copy[key]:
                        if elem["name"] == instance[0]:
                            injection_site = InjectableSite(OperatorType[key], elem["name"], elem["size"])
            # Instanties the InjectionSitesGenerator and generates a random injection with the injectable sites and
            # the campaign size.
            try:
                self.__injection_sites = InjectionSitesGenerator([injection_site],
                                                                 models_mode).generate_random_injection_sites(size)
            except:
                return False
            return True
        else:
            raise ValueError("Unrecognized mode: {}".format(mode))

    def __assign_parameters(self, session, prefix):
        """
        Copies the variables from the original session to the session given.
        By variables are meant all the weights and parameters presente in the session.

        Arguments:

            session {tf.Session} -- TensorFlow's session which will be the destination of the copy.

            prefix {str} -- Prefix of each name in the destination session's graph.
        """
        update_operations = []  # List of TensorFlow's operations regarding the copy.
        # Slides through each variable presents in the original graph.
        for variable in self.__old_graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            # Evaluates the variable to get its value.
            tensor = self.__old_session.run(variable)
            # Gets the same variable in the new session.
            new_variable = session.graph.get_tensor_by_name(prefix + variable.name)
            # Creates the update operation.
            update_operations.append(tf.assign(new_variable, tensor))
        # Executes the update operations, it may take time.
        session.run(update_operations)

    def __get_replicated_inputs(self, inputs, fault_map=None):
        """
        Given tensors from the original graph, it looks in the fault map
        to find their replicated version.

        Arguments:

            inputs {list(tf.Tensor)} -- List of TensorFlow's tensor coming from the original graph
            to be matched with their faulty versions.

        Keyword Arguments:

            fault_map {dict} -- Fault map that is used for the lookup.
            If None or avoided, it will use the default fault map.

        Returns:

            [list(tf.Tensor)] -- Returns the list of faulty tensors.
        """
        # If the fault map has not been provided, it takes the default fault map.
        if fault_map is None:
            fault_map = self.__fault_map
        replicated_inputs = []
        # Slides through each input tensor and lookup the faulty version.
        for input_tensor in inputs:
            replicated_inputs.append(fault_map[input_tensor.name])
        return replicated_inputs

    def __set_replicated_outputs(self, zipped_outputs, fault_map=None):
        """
        Sets the given pairs of original tensor and replicated tensor in the fault map.

        Arguments:

            zipped_outputs {list((tf.Tensor, tf.Tensor))} -- List of tensors pair.

        Keyword Arguments:

            fault_map {dict} -- Fault map that is used for the lookup.
            If None or avoided, it will use the default fault map.
        """
        # If the fault map has not been provided, it takes the default fault map.
        if fault_map is None:
            fault_map = self.__fault_map
        # Slides through each pair and set them into the fault map.
        for output, replicated_output in zipped_outputs:
            fault_map[output.name] = replicated_output

    def __create_operator_replica(self, graph, operator, inputs):
        """
        Creates a replica of the given operator that will belong to the given graph.

        Arguments:

            graph {tf.Graph} -- TensorFlow's graph in which will be created the operator replica.

            operator {tf.Operation} -- TensorFlow's operation to be replicated.

            inputs {list(tf.Tensor)} -- List of tensors that are the inputs of the operator.

        Returns:

            [tf.Operation] -- Returns the replicated operator.
        """
        # Creates the replica on the given graph, specifying the operator type, its inputs,
        # the input types, the output types and the name.
        return graph.create_op(
            operator.type,
            inputs,
            dtypes=[t.dtype for t in operator.outputs],
            input_types=[t.dtype for t in inputs],
            op_def=operator.op_def,
            attrs=operator.node_def.attr,
            name=operator.name
        )

    def __run(self, fetches, feed_dict={}, session=None, fault_map=None, as_is=False):
        """
        Executes the fetches with the feeds provided, if any.

        Arguments:

            fetches {list(tf.Tensor)} -- List of tensors to fetch from the session.

        Keyword Arguments:

            feed_dict {dict(tf.Tensor, object)} -- Dictionary of feeds for the inference.

            session {tf.Session} -- Session on which run the fetches. If not provided, it will be used the fault one.

            fault_map {dict(tf.Tensor, tf.Tensor)} -- Fault map containing the map among original and fault tensors.
            If not provided, the default fault map will be used.

            as_is {bool} -- Tells if the the fetches and feeds need to be converted to the replicated ones.
            If False it will convert the fetches and feeds, otherwise will run with the ones provided.

        Returns:

            Returns the executed fetches.
        """
        # Selects the default fault session if not provided.
        if session is None:
            session = self.__fault_session
        # Selects the default fault map if not provided.
        if fault_map is None:
            fault_map = self.__fault_map
        # Replicates the fetches if not "as is", otherwise uses the originals.
        if not as_is:
            replicated_fetches = self.__get_replicated_fetches(fetches, fault_map=fault_map)
        else:
            replicated_fetches = fetches
        # Replicates the feeds.
        replicated_feed_dict = self.__get_replicated_feed_dict(feed_dict, fault_map=fault_map)
        # Executes the run and returns the output.
        return session.run(replicated_fetches, feed_dict=replicated_feed_dict)  # [0]

    def __is_iterable(self, item):
        """
        Checks if the given item is iterable or not.

        Arguments:
            item {object} -- An item to test if it's iterable or not.

        Returns:
            [bool] -- Returns True if the item is iterable, otherwise False.
        """
        try:
            _ = [x for x in item]  # If not iterable, it will raise an exception.
            return True
        except:
            return False

    def __get_replicated_fetches(self, fetches, fault_map=None):
        """
        Given fetches from the original model, it returns the replicated fetches.

        Arguments:

            fetches {list, dict, tf.Tensor} -- Data structure (list, dict or single) holding the fetches.

        Keyword Arguments:

            fault_map {dict(tf.Tensor, tf.Tensor)} -- Fault map containing the map among original and fault tensors.
            If not provided, the default fault map will be used.

        Returns:
            [dict, list, tf.Tensor] -- Returns the replicated fetches.
        """
        # Selects the default fault map if not provided.
        if fault_map is None:
            fault_map = self.__fault_map
        # Checks if the fetches are iterable or not.
        if self.__is_iterable(fetches):
            # Returns the data as dict if fetches is a dict, otherwise returns a list.
            if isinstance(fetches, dict):
                return {key: fault_map[t.name] for (key, t) in fetches.items()}
            else:
                return [fault_map[fetch.name] for fetch in fetches]
        else:
            # Returns just the replicated fetch.
            return fault_map[fetches.name]

    def __get_replicated_feed_dict(self, feed_dict, fault_map=None):
        """
        Given a feed dict, returns the replicated feed dict.

        Arguments:

            feed_dict {dict} -- Model's feeds.

        Keyword Arguments:

            fault_map {dict(tf.Tensor, tf.Tensor)} -- Fault map containing the map among original and fault tensors.
            If not provided, the default fault map will be used.

        Returns:
            [dict] -- Replcated feeds.
        """
        # Selects the default fault map if not provided.
        if fault_map is None:
            fault_map = self.__fault_map
        return {
            fault_map[placeholder.name]: placeholder_value
            for (placeholder, placeholder_value) in feed_dict.items()
        }

    def __extract_injectable_sites(self, graph, operations_histogram):
        """
        Given the operators' histogram, build the injectables sites that are used by
        InjectionSitesGenerator.

        Arguments:

            graph {tf.Graph} -- The graph to which extracts the operators info.

            operations_histogram {dict(str,str)} -- Operators histogram.
        """
        injectable_sites = {}  # Map of injectables sites. The keys are the operator types,
        # The values are the operator names and sizes.
        # Slides over operator types.
        for operator_type in operations_histogram.keys():
            # Slides over operator names.
            for operator_name in operations_histogram[operator_type]:
                # Copies the operator type in this temporary variable
                # because for Conv2D is necessary to change the operator type.
                injectable_site_type = operator_type
                # Gets the operator from the graph.
                operator = graph.get_operation_by_name(operator_name)
                # assert operator.outputs[0].shape.is_fully_defined(), (operator_name, operator.outputs[0].shape)
                operator_signature = OrderedDict()  # The signature of the operator (name and size) is a dictionary.
                if operator_type == "FusedBatchNorm":
                    # injectable_site_type = "FusedBatchNorm_test"
                    injectable_site_type = "FusedBatchNorm"
                if operator_type == "Conv2D":
                    # In case of convolution, we need to extract the kernel size a
                    kernel_size = int(operator.inputs[1].shape[1])
                    input_size = int(operator.inputs[0].shape[2])
                    output_size = int(operator.outputs[0].shape[2])
                    strides = input_size / output_size
                    # According to the kernel size and the stride there are
                    # different operator type.
                    if kernel_size == 1:
                        injectable_site_type = "Conv2D1x1"
                    elif kernel_size == 3 and strides == 1:
                        injectable_site_type = "Conv2D3x3"
                        # injectable_site_type = "Conv2D_test"
                    elif kernel_size == 3 and strides == 2:
                        injectable_site_type = "Conv2D3x3S2"
                    else:
                        injectable_site_type = "Conv2D3x3S2"
                        # print(
                        # "Unsupported Conv2D configuration [K = {}, S = {}], skipping.".format(kernel_size, strides))
                        # continue
                # Sets the name and the size of the operation.
                operator_signature["name"] = operator_name
                operator_signature["size"] = str(operator.outputs[0].shape)
                # Adds the operation signature according the type.
                if injectable_site_type not in injectable_sites:
                    injectable_sites[injectable_site_type] = []
                injectable_sites[injectable_site_type].append(operator_signature)
        # Sets the injectables sites.
        self.__injectable_sites = injectable_sites

    def inject(self, fetches, feeds, callback=None):
        """

        Performs the injections, after having instrumented the model and
        generated the injection sites.

        Arguments:

            fetches {list, dict, tf.Tensor} -- TensorFlow's fetches.

            feeds {dict} -- TensorFlow's feeds.
        """
        start_time = time.time()
        # Obtains the replicated fetches and feeds.
        replicated_fetches = self.__get_replicated_fetches(fetches)
        replicated_feeds = self.__get_replicated_feed_dict(feeds)
        delta_t1 = 0
        delta_t2 = 0
        delta_t3 = 0
        cache = {}
        count = 0
        results = defaultdict(list)
        # Slides over injection sites:
        for injection_site in self.__injection_sites:
            # Gets the fault point operator.
            print("Injecting: " + injection_site.operator_name)
            fault_fetches = self.__fault_fetches[injection_site.operator_name]
            # Runs the up to the injectable operator.
            # t1 = time.time()
            # partial_output = self.__fault_session.run(fault_fetches, feed_dict=replicated_feeds)[0]
            # t2 = time.time()
            # Injects each value at the specific index.
            # output_dimensions = len(partial_output.shape)
            if injection_site.operator_name not in cache:
                # fault_fetches = self.__fault_fetches[injection_site.operator_name]
                cache[injection_site.operator_name] = \
                    self.__fault_session.run(fault_fetches, feed_dict=replicated_feeds)[0]
            partial_output = np.copy(cache[injection_site.operator_name])
            for index, value in injection_site:
                # If the value type is [-1, 1] then the raw value has to be added at the current value.
                # if len(index) != output_dimensions:
                #    index = tuple([index[i] for i in range(len(index) - output_dimensions, len(index))])
                # print(index, value.value_type, value.raw_value)
                if value.value_type == "[-1,1]":
                    partial_output[index] += value.raw_value
                else:
                    # Otherwise, set the value itself.
                    partial_output[index] = value.raw_value
            # Creates a dirty feed with the partial output.
            dirty_feeds = {tensor: value for (tensor, value) in zip(fault_fetches, [partial_output])}
            dirty_feeds.update(replicated_feeds)
            # Gets the output with the fault tensor.
            t3 = time.time()
            outputs = self.__fault_session.run(replicated_fetches, feed_dict=dirty_feeds)
            for i in range(len(fetches)):
                results[i].append(outputs[i])
            if callback is not None:
                if callback(outputs):
                    count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        average_injection_time = elapsed_time / float(len(self.__injection_sites))
        print("Total elapsed time: {:.5f} s for {} injections.".format(elapsed_time, len(self.__injection_sites)))
        print("Average injection time: {:.5f} s.".format(average_injection_time))

        return results, average_injection_time, self.__injection_sites

    def fake_inject(self, fetches, feeds, size):
        start_time = time.time()
        # Obtains the replicated fetches and feeds.
        replicated_fetches = self.__get_replicated_fetches(fetches)
        replicated_feeds = self.__get_replicated_feed_dict(feeds)
        operator_names = []
        global_outputs = []
        for item in self.__injectable_sites.items():
            for item_2 in item[1]:
                for item_3 in item_2.items():
                    if item_3[0] == "name":
                        operator_names.append(item_3[1])
        for _ in xrange(size):
            random_operator_index = np.random.randint(len(operator_names))
            fault_fetches = self.__fault_fetches[operator_names[random_operator_index]]
            partial_output = self.__fault_session.run(fault_fetches, feed_dict=replicated_feeds)[0]
            modified_output = np.copy(partial_output)
            dirty_feeds = {tensor: value for (tensor, value) in zip(fault_fetches, [modified_output])}
            dirty_feeds.update(replicated_feeds)
            outputs = self.__fault_session.run(replicated_fetches, feed_dict=dirty_feeds)
            global_outputs.append(outputs)
        end_time = time.time()
        print("Elapsed time for a campaign of size {}: {}".format(size, end_time - start_time))
        return global_outputs

    def __del__(self):
        # Close the TensorFlow's session.
        if self.__fault_session is not None:
            self.__fault_session.close()

    instrumentation_done = property(lambda self: self.__instrumentation_done)
