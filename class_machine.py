# Date: 01/04/2019
# Auth: Manuel Cremades, manuel.cremades@usc.es

# Basic modules
import sys; sys.path.insert(0,'..'); from fatDAE.base.basic_import import *

class Machine(object):
    ''' State machine.
    '''

    def print_states(self):

        for event in self.events:
            print('event ->', event)

        for state in self.states:
            print('state ->', state)

            for transition in state.transitions:

                print('transition ->', transition)

                print('ini_state ->',transition.ini_state)
                print('end_state ->',transition.fin_state)

                for event in transition.events:
                    print('event ->', event)

    def __init__(self, max_number_states=1, print_level=0):

        self.states = []

        self.actual_state = None; self.prev_state = None
        self.print_level = print_level
        self.events = []

        self.max_number_states = max_number_states

        self.number_states_total = 0
        self.number_states_count = 0

    def check(self, params):
        ''' Checks the actual state event. If the event is accepted then jumps to next state.

        Returns:
            (:obj:`bool`)
        '''

        params['state_t'] = self.actual_state.t

        params['number_states_total'] = self.number_states_total
        params['number_states_count'] = self.number_states_count

        x, h, event, accept, k = self.actual_state.check(params)

        if event:

            #print('TRIGED')
            if accept:
                #print('Acepted')
                #print(self.actual_state.next_state)
                #self.print_states()

                self.prev_state = self.actual_state
                self.prev_state.exec_out(params, reset=self.prev_state.transitions[k].reset)
                print(self.prev_state.params)
                self.actual_state = self.actual_state.next_state
                
                self.actual_state.exec_ini(params)

                self.number_states_total += 1
                self.number_states_count += 1


        return x, h, event, accept

    def add_states(self, states):
        ''' Appends a list of states to the current list and the associated transitions.

        Args:
            states (:obj:`list`): List of :obj:`state_machine.class_state_machine.State` instances.
        '''

        for state in states:

            self.states.append(state)

            for transition in state.transitions:

                self.events.extend(transition.events)

class State(object):
    ''' State of operation.
    '''

    def __init__(self, name='', print_level=0):

        self.name = name
        self.print_level = print_level

        self.transitions = []

        self.next_state = None

        self.params = {}

        self.params['number_states_count']=0
        self.params['number_states_total']=0

    def plot(self, save_path=None):
        pass

    def exec_ini(self, params):
        ''' To be executed when the state is changed.
        '''
        self.params['number_states_count']+=1
        self.params['number_states_total']+=1
        pass

    def exec_dur(self, params, accept=False):
        ''' To be executed every time :meth:`check` is called.
        '''
        pass

    def exec_out(self, params, reset=False):
        ''' To be executed when the state is changed.
        '''
        if reset:
            print('Resetting count')
            self.params['number_states_count']=0
        pass

    def store(self, params):
        pass

    def check(self, params):
        ''' Checks if the state must change by checking its transitions, and performs the change of state.

        Returns:
            (tuple): Tuple containing:

            - (:obj:`bool`): True if some event is trigered, False otherwise.
            - (:obj:`bool`): True if some event is accepted, False otherwise.

        '''

        params['state_params'] = self.params

        self.exec_dur(params)

        trigered_list = []
        accepted_list = []

        jump_list = []
        jump_step = []

        x_list = []
        h_list = []

        i = 0
        for transition in self.transitions:

            x, h, trigered, accepted = transition.check(params)

            if True in trigered:
                trigered_list.append(True)
            else:
                trigered_list.append(False)

            if False in accepted:
                accepted_list.append(False)
            else:
                accepted_list.append(True)

            if True in trigered and True in accepted:
                jump_list.append(i)
                jump_step.append(min(h))
            i += 1

            x_list.append(x)
            h_list.append(h)

        if True in trigered_list:
            trigered = True
        else:
            trigered = False

        if False in accepted_list:
            accepted = False
        else:
            accepted = True

        if len(jump_list) > 0:

            k = numpy.argmin(numpy.array(jump_step))
            self.exec_out(params, self.transitions[jump_list[k]].reset)
            
            self.next_state = self.transitions[jump_list[k]].fin_state

            return x, jump_step[k], trigered, accepted, k
        return x, min(h_list)[0], trigered, accepted, 0


    def add_transitions(self, transitions):
        ''' Appends a list of transitions to the current list.
        '''

        self.transitions.append(transitions)

class End(State):
    pass

class Transition(object):
    ''' Transition between states.

    Args:
        ini_state (:obj:`state_machine.class_state_machine.State):
        fin_state (:obj:`state_machine.class_state_machine.State):
    '''

    def __init__(self, ini_state, fin_state, reset=0, print_level=0):

        self.events = []

        self.reset = reset
        self.print_level = print_level
        self.ini_state = ini_state
        self.fin_state = fin_state

    def check(self, params):
        ''' Checks if the transitions happens by checking its triggers.

        Returns:
            (tuple): Tuple containing:

            - (:obj:`bool`): True if some trigger is trigered, False otherwise.
            - (:obj:`bool`): True if some trigger is accepted, False otherwise.

        '''

        x_list = []
        h_list = []

        trigered_list = []
        accepted_list = []

        for event in self.events:

            x, h, trigered, accepted = event.check(params)

            x_list.append(x)
            h_list.append(h)

            trigered_list.append(trigered)
            accepted_list.append(accepted)

        return x_list, h_list, trigered_list, accepted_list

    def add_events(self, events):
        ''' Appends a list of events to the current list.

        Args:
            states (:obj:`list`): List of :obj:`state_machine.class_state_machine.Event` instances.
        '''

        self.events.append(events)

class Event(object):
    ''' Event of a transition.
    '''

    def __init__(self, print_level=0):
        self.print_level = print_level

    def check(self, params):
        pass

    def f(self):
        pass

class MaxCycles(Event):

    def __init__(self, n=1):

        Event.__init__(self)

        self.n = n

    def check(self, params):

        x_k = params['x_k']
        h_k = params['h_k']

        if params['state_params']['number_states_count'] > self.n:

            if self.print_level > -1:
                print('Event located')

            return x_k, h_k, True, True

        else:

            return x_k, h_k, False, True

class Wait(Event):
    ''' Event of a transition.
    '''

    def __init__(self, t, tol_a=1e-4, tol_r=1e-2, print_level=0):

        Event.__init__(self, print_level=print_level)

        self.t = t

        self.tol_a = tol_a
        self.tol_r = tol_r

    def check(self, params):

        problem = params['problem']

        x_0 = params['x_0']
        x_k = params['x_k']
        t_0 = params['t_0']
        h_k = params['h_k']

        t_k = t_0 + h_k

        if abs((params['state_t'] + self.t) - t_k) / self.tol_a < 1:

            if self.print_level > -1:
                print('Event located')

            return x_k, h_k, True, True

        else:

            if (params['state_t'] + self.t) - t_k < 0.0:
                
                if self.print_level > -1:
                    print('Locating event... ')     

                h_k = (params['state_t'] + self.t) - t_0

                return x_k, h_k, True, False

            else:

                return x_k, h_k, False, True
