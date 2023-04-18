# Date: 01/04/2019
# Auth: Manuel Cremades, manuel.cremades@usc.es

import numpy

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

    def __init__(self, max_number_states=1):

        self.states = []; self.actual_state = None

        self.events = []

        self.max_number_states = max_number_states;

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

        x, h, event, accept = self.actual_state.check(params)

        if event:

            #print('TRIGED')
            if accept:
                #print('Acepted')
                #print(self.actual_state.next_state)
                #self.print_states()

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

    def __init__(self, name=''):

        self.name = name

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

    def exec_out(self, params):
        ''' To be executed when the state is changed.
        '''
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

            #trigered_list.append(trigered)
            #accepted_list.append(accepted)

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

            #print('th',h)

        #print(trigered_list)
        #print(accepted_list)

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

            return x, jump_step[k], trigered, accepted
        return x, min(h_list)[0], trigered, accepted

        #i_trigered = numpy.where(numpy.array(trigered_list))[0]
        #i_accepted = numpy.where(numpy.array(accepted_list))[0]

        #trigered = False
        #accepted = True

        #if [True] in trigered_list:
        #    trigered = True
        #if [False] in accepted_list:
        #    accepted = False

        #if len(jump_list) > 0:

        #    print('jump list', jump_list)

        #    ind = numpy.argmin(jump_list)

        #    self.exec_out(params, self.transitions[jump_list[ind]].reset); self.next_state = self.transitions[jump_list[ind]].fin_state

        #    trigered = True
        #    accepted = True

        #if i_trigered.size > 0:

        #    trigered = True

        #    print('trig',i_trigered)
        #    print('acep',i_accepted)

        #    if i_trigered[0] in i_accepted:

        #        self.exec_out(params, self.transitions[i_trigered[0]].reset)

        #        self.next_state = self.transitions[i_trigered[0]].fin_state

        #        accepted = True

        #    else:
        #        accepted = False

        #else:

        #    if i_accepted.size > 0:

        #        accepted = True

        #print('check state', trigered, accepted)

        #return x, min(h_list), trigered, accepted

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

    def __init__(self, ini_state, fin_state, reset=0):

        self.events = []

        self.reset = reset

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

            #print('event h',h)

            x_list.append(x)
            h_list.append(h)

            trigered_list.append(trigered)
            accepted_list.append(accepted)

        #print(h_list)

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

    def __init__(self):
        pass

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

            print('Event located')

            return x_k, h_k, True, True

        else:

            return x_k, h_k, False, True

class Wait(Event):
    ''' Event of a transition.
    '''

    def __init__(self, t, tol_a=1e-4, tol_r=1e-2):

        Event.__init__(self)

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

            print('Event located')

            return x_k, h_k, True, True

        else:

            if (params['state_t'] + self.t) - t_k < 0.0:

                print('Locating event... ')

                h_k = (params['state_t'] + self.t) - t_0

                #print('h_k',h_k)

                return x_k, h_k, True, False

            else:

                return x_k, h_k, False, True
