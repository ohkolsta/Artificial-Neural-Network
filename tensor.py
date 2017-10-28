import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
from random import uniform, sample
from mnist_basics import load_all_flat_cases

#hide redundant logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#used file tutor3.py provided in course IT3105 Artifical Programming as basis
#tflowtools.py provided in couse as well

class Gann():

    def __init__(self,dims,num_inputs,activation_hidden,activation_output,cost_function,init_weight_range,cman,lrate=.1,showint=None,mbs=10,vint=None):
        self.num_inputs = num_inputs
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.cost_function = cost_function
        self.init_weight_range = init_weight_range

        self.learning_rate = lrate
        self.layer_sizes = dims # Sizes of each layer of neurons
        self.show_interval = showint # Frequency of showing grabbed variables
        self.global_training_step = 0 # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.grabvar_figures = [] # One matplotlib figure for each grabvar
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.caseman = cman
        self.modules = []
        self.build()

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type,spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self,module_index,type='wgt',add_figure=True):
        self.grabvars.append(self.modules[module_index].getvar(type))
        if add_figure:
            self.grabvar_figures.append(PLT.figure())

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self,module):
        self.modules.append(module)

    def build(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]

        #returns activation output from user-chosen activation function
        def gen_act_output(output):
            function = 'tf.nn.%s(output)' % (self.activation_output)
            return eval(function)

        num_inputs = self.num_inputs
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')

        invar = self.input; insize = num_inputs
        # Build all of the modules
        for i,outsize in enumerate(self.layer_sizes[1:]):
            gmod = Gannmodule(self,self.activation_hidden,self.init_weight_range,i,invar,insize,outsize)
            invar = gmod.output; insize = gmod.outsize
        self.output = gmod.output # Output of last module is output of whole network
        if self.activation_output: self.output = gen_act_output(self.output)
        self.target = tf.placeholder(tf.float64,shape=(None,gmod.outsize),name='Target')
        self.configure_learning()

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self):

        #returns cost from user_chosen cost function
        def get_cost(cost_function, logits, labels):
            function = 'tf.nn.%s(logits=logits, labels=labels)' % (self.cost_function)
            return tf.reduce_mean(eval(function))

        if self.cost_function:
            cross_entropy = get_cost(self.cost_function,logits=self.output,labels=self.target)
            self.error = tf.reduce_mean(cross_entropy)
        else:
            self.error = tf.reduce_mean(tf.square(self.target - self.output),name='MSE') #original cost function
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error,name='Backprop')

    def do_training(self,sess,cases,epochs=100,continued=False):
        if not(continued): self.error_history = []
        for i in range(epochs):
            if (i % 10 == 0):
                print ("step:",i)
            error = 0; step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            mbs = self.minibatch_size; ncases = len(cases); nmb = math.ceil(ncases/mbs)
            for cstart in range(0,ncases,mbs):  # Loop through cases, one minibatch at a time.
                cend = min(ncases,cstart+mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]; targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                _,grabvals,_ = self.run_one_step([self.trainer],gvars,self.probes,session=sess,
                                         feed_dict=feeder,step=step,show_interval=self.show_interval)
                error += grabvals[0]
            #store values for post-run visualizations
            self.consider_validation_testing(step, sess)
            self.error_history.append((step, error/nmb))
        self.global_training_step += epochs

    def do_testing(self,sess,cases,msg='Testing',printing=None,bestk=None):
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor,[TFT.one_hot_to_int(list(v)) for v in targets],k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                           feed_dict=feeder,  show_interval=None)
        if printing:
            if bestk is not None:
                print ('%s Set Top_K = %f %%' % (msg,100*(testres/len(cases))))
            else:
                print('%s Set Accuracy = %f ' % (msg, 100-testres))
        return testres  # self.error uses MSE, so this is a per-case value

    #return correct matches
    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self,epochs,sess=None,dir="probeview",continued=False):
        self.roundup_probes()
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session,self.caseman.get_training_cases(),epochs,continued=continued)

    def testing_session(self,sess,bestk=None):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess,cases,msg='Final Testing',printing=True,bestk=bestk)

    def consider_validation_testing(self,epoch,sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess,cases,msg='Validation Testing',printing=False)
                self.validation_history.append((epoch,error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self,sess,bestk=None):
        self.do_testing(sess,self.caseman.get_training_cases(),msg='Total Training',printing=True,bestk=bestk)

    # Similar to the "quickrun" functions used earlier.
    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step+1)
        return results[0], results[1], sess

    #display grabbed variables
    def display_grabvars(self, grabbed_vals, grabbed_vars,step=1,):
        try:
            names = [x.name for x in grabbed_vars];
        except Exception as e:
            names = [grabbed_vars]
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                TFT.hinton_plot(v,fig=self.grabvar_figures[fig_index],title= names[i]+ ' at step '+ str(step))
                fig_index += 1
            else:
                v = np.array([v])
                if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                    TFT.hinton_plot(v,fig=self.grabvar_figures[fig_index],title= names[i]+ ' at step '+ str(step))
                    fig_index += 1

    def run(self,epochs,sess=None,continued=False,bestk=None):
        PLT.ion()
        print ('Training...')
        self.training_session(epochs,sess=sess,continued=continued)
        print ('done.')
        print ('Testing...')
        self.test_on_trains(sess=self.current_session,bestk=bestk)
        self.testing_session(sess=self.current_session,bestk=bestk)
        self.close_current_session()
        PLT.ioff()

    def runmore(self,epochs,bestk=None):
        self.reopen_current_session()
        self.run(epochs,sess=self.current_session,continued=True,bestk=bestk)

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=True)


# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule():

    def __init__(self,ann,activation_hidden,init_weight_range,index,invariable,insize,outsize):
        self.activation_hidden = activation_hidden
        self.init_weight_range = init_weight_range

        self.ann = ann
        self.insize = insize  # Number of neurons feeding into this module
        self.outsize = outsize # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Layer-"+str(self.index)
        self.build()

    def gen_act_hid_output(self, loss, name):
        function = 'tf.nn.%s(loss, name)' % (self.activation_hidden)
        return eval(function)

    def build(self):
        mona = self.name; n = self.outsize
        #set random weight according to init_weight_range boundaries
        self.weights = tf.Variable(np.random.uniform(self.init_weight_range[0],self.init_weight_range[1],
                                                        size=(self.insize,n)), name=mona+'wgt', trainable=True)
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n),
                                  name=mona+'-bias', trainable=True)  # First bias vector
        self.output = self.gen_act_hid_output(tf.matmul(self.input,self.weights)+self.biases,name=mona+'-out')
        self.ann.add_module(self)


    def getvar(self,type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self,type,spec):
        var = self.getvar(type)
        base = self.name +'_'+type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)

# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class Caseman():

    def __init__(self,cfunc,vfrac=0,tfrac=0):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca) # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases)*self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases

#return cases
def get_case(case_name, parameters, cfrac):
    if case_name == 'mnist':
        return (lambda : get_mnist_cases(cfrac))
    if case_name == 'winequality_red.txt':
        return (lambda : read_file(case_name, ';', cfrac))
    elif case_name == 'yeast.txt' or case_name == 'glass.txt' or case_name == 'pima-indians-diabetes.txt':
        return (lambda : read_file(case_name, ',', cfrac))
    else:
        paras = ''
        if len(parameters) > 1:
            for par in parameters[:-1]:
                paras += str(par) + ','
        paras += str(parameters[-1])
        function = 'TFT.gen_%s_cases(%s)' % (case_name, paras)
        return (lambda : eval(function))

#return cases from text files
def read_file(filename, delimiter, cfrac):
    textfile = np.loadtxt('data/'+filename, delimiter=delimiter)
    cases = []
    target_size = 0
    for line in textfile:
        if line[-1] > target_size - 1:
            target_size = int(line[-1]) + 1
    for line in textfile:
        target = [0] * target_size
        target[int(line[len(line)-1])] = 1
        cases.append([(line[:len(line)-1]), target])

    #normalize data
    for i in range(len(cases[0][0])):
        fmax = float('-inf')
        fmin = float('inf')
        for c in cases:
            if c[0][i] < fmin:
                fmin = c[0][i]
            if c[0][i] > fmax:
                fmax = c[0][i]
        diff = fmax - fmin
        for case in cases:
            case[0][i] = (case[0][i] - fmin) / diff

    if cfrac:
        cases = sample(cases, math.ceil(len(cases)*cfrac))
    return cases

#return MNIST cases
def get_mnist_cases(cfrac):
    inputs, targets_vector = load_all_flat_cases('training','')
    target_size = max(targets_vector) + 1

    targets = []
    for target_value in targets_vector:
        target = [0] * target_size
        target[target_value - 1] = 1
        targets.append(target)

    cases = [[inputs[i],targets[i]] for i in range(len(inputs))]

    if cfrac:
        cases = sample(cases, math.ceil(len(cases)*cfrac))
    return cases

#display error_history and validation_history in same plot post-run
def display_error_plot(error_history, validation_history):
    x_error = []
    y_error = []
    for point in error_history:
        x_error.append(point[0])
        y_error.append(point[1])
    x_validation = []
    y_validation = []
    for point in validation_history:
        x_validation.append(point[0])
        y_validation.append(point[1])
    PLT.plot(x_error, y_error)
    PLT.plot(x_validation, y_validation)

    PLT.ylabel('Error')
    PLT.xlabel('Minibatches')
    PLT.legend(['error history', 'validation history'], loc='upper right')
    PLT.title('Error progression')
    PLT.show(block=False)

#map cases by running them through the network without learning
def do_mapping(self,sess,cases,msg='Mapping',bestk=1):
    inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
    feeder = {self.input: inputs, self.target: targets}
    self.test_func = self.error
    if bestk is not None:
        self.test_func = self.gen_match_counter(self.predictor,[TFT.one_hot_to_int(list(v)) for v in targets],k=bestk)
    error, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                       feed_dict=feeder,  show_interval=None)

    return grabvals  # self.error uses MSE, so this is a per-case value

#map cases to evaluate network
def mapping(self,all_cases,mapbs,map_layers,map_dendrograms):
    self.reopen_current_session()
    print ('Mapping starts...')

    mapping_cases = sample(all_cases, mapbs)
    self.grabvars = []
    for layer in map_layers:
        self.add_grabvar(layer,'in',add_figure=False)
        self.add_grabvar(layer,'out',add_figure=False)
        self.add_grabvar(layer,'wgt',add_figure=False)
        self.add_grabvar(layer,'bias',add_figure=False)

    mapping_grabvals = []
    mapping_biases = []
    for case in mapping_cases:
        cur_cases = [mapping_cases[0], case] #TODO find a way to only use one case
        mapping = do_mapping(self,self.current_session,cur_cases,'Mapping')
        mapping_grabvals.append(mapping)
        if len(mapping) > 1:
            mapping_biases.append(mapping[1])

    all_module_vals = []
    for case in mapping_grabvals:
        n = 4 #split on four variables: input,output,weights,biases
        module_vals = []
        for module in [case[i:i+n] for i in range(0,len(case),n)]:
            input = module[0][1]
            output = module[1][1]
            weights = module[2]
            biases = module[3]
            module_vals.append([input,output,weights,biases])
        all_module_vals.append(module_vals)

    #visualization of activation levels (output) for each case in each layer
    saved_m_index = 0 #mapping only level 1 places values on index 0
    for module_index in map_layers:
        case_number = 0
        for case in all_module_vals:
            output = case[saved_m_index][1]
            print ("displaying mapping activation levels...")
            output = np.array([output])
            TFT.hinton_plot(output,fig=PLT.figure(),title='Activation level module '+str(module_index)+': case: '+str(case_number))
            case_number += 1
        saved_m_index += 1

    #making dendrograms
    if len(map_dendrograms) > len(map_layers):
        raise ValueError('Custom error: Layers for dendrograms must be mapped via map_layers (len(map_dendrograms) > len(map_layers))')
    saved_m_index = 0
    for module_index in map_dendrograms:
        all_inputs = []
        all_outputs = []
        for case in all_module_vals:
            all_inputs.append(case[saved_m_index][0])
            all_outputs.append(case[saved_m_index][1])
        title = "Dendrogram module %d: output" % module_index
        print ("displaying dendrograms...")
        TFT.dendrogram(all_outputs, None, title=title)
        PLT.pause(1)
        saved_m_index += 1

#   ****  MAIN functions ****
# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.
def autoex(
            dims=[8,4,8],
            activation_hidden='relu',
            activation_output='softmax',
            cost_function='softmax_cross_entropy_with_logits',
            lrate=0.25,
            init_weight_range=[-.1,.1],
            data_source=['glass.txt', [0]],
            cfrac=1,
            vfrac=0.1,
            tfrac=0.1,
            mbs=100,
            mapbs=10,
            steps=300,
            map_layers=[0,1],
            map_dendrograms=[0,1],
            display_weights=None,
            display_biases=None,
            display_vis=True,
            bestk=1):

    #generate cases and set network
    case_generator = get_case(data_source[0], data_source[1], cfrac)
    showint = steps-1 #only show grabbed variables after all steps
    vint = 1 #validate for every step
    cman = Caseman(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac)
    num_inputs = len(cman.training_cases[0][0])
    size = len(cman.training_cases[0][1])
    if not mbs:
        mbs = size
    ann = Gann(dims=dims,num_inputs=num_inputs,activation_hidden=activation_hidden,
                activation_output=activation_output,cost_function=cost_function,
                init_weight_range=init_weight_range,cman=cman,lrate=lrate,
                showint=showint,mbs=mbs,vint=vint)

    #grab weights
    for layer in display_weights:
        ann.add_grabvar(layer, 'wgt')

    #grab biases
    for layer in display_biases:
        ann.add_grabvar(layer, 'bias')

    #run network
    ann.run(steps,bestk=bestk)

    #display visualizations
    if display_vis:
        PLT.figure()
        display_error_plot(ann.error_history, ann.validation_history)
        if mapbs != 0:
            mapping(ann,cman.cases,mapbs,map_layers,map_dendrograms)

if __name__ == "__main__":
    autoex(
                dims=[784,1568,10],
                activation_hidden='relu',
                activation_output=None,
                cost_function='softmax_cross_entropy_with_logits',
                lrate=0.5,
                init_weight_range=[-.1,.1],
                data_source=['mnist', [0]],
                cfrac=0.1,
                vfrac=0.1,
                tfrac=0.1,
                mbs=200,
                mapbs=10,
                steps=10,
                map_layers=[],
                map_dendrograms=[],
                display_weights=[0],
                display_biases=[1],
                display_vis=True,
                bestk=1)
    print ("Done.")
    PLT.show()
