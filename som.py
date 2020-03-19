from typing import NamedTuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

'''
Notes for reader:

* I have prefixed comments that are related to the challenge itself with
"Author's note:". I have otherwise strived to capture all notable details
of the implementation in docstrings and inline comments.

* I have used the convention of prefixing identifiers with an underscore to
communicate to users that the identified artefacts are intended to be private
to the class. I am aware of Python's double prefix mechanism but prefer to not
be so prescriptive.

* I have loosely followed the pep8 code style conventions and the Google
convention for docstrings.

* There are no tests for this code due to lack of time.

* If you spot any mistakes or areas for improvement please let me know :)

Guide:

* Class SelfOrganisingMap is the core map implementation.
* Class TrainingRegime generates the progression of training parameters.
* Function create_som manages the interactions between the above classes.
* Class Animator is a convenience class for generating map animations.
'''

# Raise select warnings to errors to avoid masking issues.
np.seterr(divide='raise', over='raise', invalid='raise')


class SelfOrganisingMap:
    '''
    Kohonen self-organising map. Contains the map, positions of the nodes
    and methods for fitting the map to an input vector. Intended to be used
    in conjunction with the TrainingRegime class.
    '''

    def __init__(self, rng, width, height, dimensions, threshold_nbhood=True):
        '''
        Initialises the map. The map's nodes' weights are initialised randomly
        using rng. If threshold_nbhood is set to True, nodes further than the
        radius of the neighbourhood from the best matching unit are strictly
        not updated (as opposed to the interpretation of decayed influence over
        distance as per the neighbourhood function implicitly excluding nodes).

        Args:
            rng: the random number generator for weight initialisation
            width: width of the grid in number of nodes
            height: height of the grid in number of nodes
            dimensions: the dimensionality of the map's weights
            threshold_nbhood: whether to threshold the neighbourhood

        Raises:
            ValueError: if any arguments are invalid
        '''
        if width <= 0:
            raise ValueError('\'width\' must be greater than zero')

        if height <= 0:
            raise ValueError('\'height\' must be greater than zero')

        # Prevent the creation of maps that would cause the initialisation
        # of TrainingRegime to fail. The smallest map radius that won't
        # cause a divide by zero in the calculation of the time constant
        # for training is 1.5.
        self._radius = max(width, height) / 2
        if self._radius < 1.5:
            raise ValueError('Map must have a radius of at least 1.5')

        if dimensions < 0:
            raise ValueError('\'dimensions\' must be greater than zero')

        # Randomly initialise node weights.
        self._nodes = rng.random((width, height, dimensions))

        self._threshold_nbhood = threshold_nbhood

    @property
    def radius(self):
        '''
        Returns:
            'Radius' of the map as per the training algorithm.
        '''
        return self._radius

    @property
    def nodes(self):
        '''
        Returns:
            Array of node weights.
        '''
        # A copy of the array is returned to avoid direct changes to
        # the map originating outside of the SelfOrganisingMap object.
        return self._nodes.copy()

    def best_matching_unit(self, vector):
        '''
        Returns:
            Index of the best matching unit for the given vector.
        '''
        return self._smallest_vector(vector - self._nodes)

    @staticmethod
    def _smallest_vector(vector_diff):
        '''
        Returns the index of the smallest vector in the given array. This is
        useful for determining the best matching unit when the array contains
        the difference between node weights and an input vector.

        Args:
            vector_diff: array representing the difference between nodes'
                weights and an input vector

        Returns:
            Index of the best matching unit node.
        '''
        vector_dist = np.apply_along_axis(np.linalg.norm, 2, vector_diff)
        return np.unravel_index(np.argmin(vector_dist), vector_diff.shape[:2])

    def update(self, iteration, vector):
        '''
        Calculates the weight changes that fit the map to the given vector
        according to the constraints of the current iteration and updates
        the map accordingly.

        Args:
            vector: input vector on which to fit the map
            iteration: details of the current iteration
        '''
        # Note that Python ignores assert statements in optimised mode. In the
        # context of training, in which this method is repeatedly called, let
        # the method explode in optimised mode trusting that the user did the
        # right thing rather than incur the repeated cost of this check.
        assert vector.shape == (self._nodes.shape[2],),\
            '\'vector\' dimensionality must match maps\' weight dimensionality'

        # Calculate difference between weights and input vector.
        vector_diff = vector - self._nodes

        # Determine the best matching unit from the difference vectors.
        bmu_index = self._smallest_vector(vector_diff)

        # Determine the distances from the best matching unit to other nodes.
        x, y = np.ogrid[0:self._nodes.shape[0], 0:self._nodes.shape[1]]
        bmu_dist = np.sqrt((x-bmu_index[0])**2+(y-bmu_index[1])**2)

        # Calculate influence array and make it the same shape as vector_diff.
        influence = iteration.influence(bmu_dist)[:,:, np.newaxis]

        if self._threshold_nbhood:
            # Override the influence on nodes outside the neighbourhood.
            influence[bmu_dist>iteration.nbhood_radius] = 0

        # Scale the update by the learning rate and influence.
        self._nodes += vector_diff * influence * iteration.learning_rate


class TrainingRegime:
    '''
    Represents a prescribed training regime of a finite number of iterations
    with 'time' (iteration) varying training parameters. TrainingRegime objects
    are intended to be iterated over, yielding Iterations that provide the
    learning rate, neighbourhood radius and influence function required
    for self-organising map training.

    Author's note: The training regime, although motivated by the self-
    organising map, is functionally independent. By using separate classes
    for the map artefact and the training regime both can be understood in
    isolation, assisting maintenance and testing.
    '''

    def __init__(
        self,
        max_iterations,
        initial_nbhood_radius,
        initial_learning_rate=0.1,
    ):
        '''
        Validates and stores training hyperparameters.

        Args:
            max_iterations: number of iterations in the training regime
            initial_nbhood_radius: initial neighbourhood radius
            initial_learning_rate: initial learning rate

        Raises:
            ValueError: if any arguments are invalid
        '''
        if max_iterations <= 0:
            raise ValueError('\'max_iterations\' must be greater than zero')

        if initial_nbhood_radius <= 1:
            raise ValueError('\'initial_nbhood_radius\' must be greater than one')

        if not (0 < initial_learning_rate <= 1):
            raise ValueError('\'initial_learning_rate\' must be in range (0,1]')

        self._max_iterations = max_iterations
        self._initial_nbhood_radius = initial_nbhood_radius
        self._initial_learning_rate = initial_learning_rate

        # Compute the unchanging time constant now.
        self._time_constant = max_iterations / np.log(initial_nbhood_radius)

    @property
    def max_iterations(self):
        '''
        Returns:
            The number of iterations in this training regime.
        '''
        return self._max_iterations

    def __iter__(self):
        '''
        Yields sequential Iteration values that collectively make up the
        training regime. Includes the initial iteration for reporting
        purposes. Intended to be called once per iteration of self-organising
        map training.

        Yields:
            Iterations values from the initial up to the max (inclusive).
        '''
        for iteration in range(0, self.max_iterations+1):
            time_factor = np.exp(-iteration / self._time_constant)
            learning_rate = self._initial_learning_rate * time_factor
            nbhood_radius = self._initial_nbhood_radius * time_factor
            dist_factor = 2 * nbhood_radius**2

            yield Iteration(
                iteration,
                learning_rate,
                nbhood_radius,
                dist_factor
            )


class Iteration(NamedTuple):
    '''
    State of an iteration within the self-organising map training regime.

    Author's note: I find representing the state of an iteration in one value
    (a named tuple) convenient as it keeps the state in one place and keeps
    the state immutable (reducing the risk of errors due to unintentional
    mutation during maintenance). Note that dist_factor is only a field for
    the avoidance of calculating the denominator in the neighbourhood influence
    function for each node.

    Fields:
        number: the number of the iteration (for progress reporting purposes)
        learning_rate: the 'current' learning rate
        nbhood_radius: the 'current' neighbourhood radius
        dist_factor: the denominator in the influence calculation
            (see Iteration.influence)
    '''
    number: int
    learning_rate: float
    nbhood_radius: float
    # Denominator of the neighbourhood influence function. This is a field
    # purely to avoid recalculating this for each invocation of the influence
    # method.
    dist_factor: float

    def influence(self, distance):
        '''
        Returns the attenuated influence 'distance' units away from the source
        of influence. In the context of self-organising maps, this is the
        distance of a node from the best matching unit.

        Args:
            distance: the distance between the entities

        Returns:
            A 'quantity' of influence attenuated over distance
        '''
        return np.exp(-distance**2 / self.dist_factor)


def create_som(
    rng,
    input_vectors,
    som_kwargs,
    regime_kwargs,
    threshold_nbhood,
    callback=None
):
    '''
    Creates and fits a self-organising map to the given input data according
    to the specified training regime. This is a convenience function to avoid
    the need to create a map and manage its training directly. At the end of
    each iteration, 'callback' is executed. This is an opt-in mechanism to
    allow for progress reporting.

    Args:
        rng: the random number generator to use
        input_vectors: array of training examples on which to fit the map
        som_kwargs: width and height keyword arguments for map
        regime_kwargs: keyword arguments for the training regime
            (see TrainingRegime.__init__), except initial_nbhood_radius
            which is determined from the map's radius
        threshold_nbhood: whether to threshold the neighbourhood function
        callback: function to call at the conclusion of each iteration

    Returns:
        Self-organising map trained on the given data.

    Raises:
        ValueError: if any arguments are invalid
    '''
    if len(input_vectors.shape) < 2:
        raise ValueError('\'input_vectors\' must have shape (x, y)')

    if input_vectors.shape[1] <= 0:
        raise ValueError('\'input_vectors\' must contain examples')

    # Create the map to be trained.
    som = SelfOrganisingMap(
        rng,
        dimensions=input_vectors.shape[1],
        threshold_nbhood=threshold_nbhood,
        **som_kwargs
    )

    # Define the progression of training parameters over the training process.
    regime = TrainingRegime(initial_nbhood_radius=som.radius, **regime_kwargs)

    # Specify a no-op callback if none was supplied to avoid branching.
    callback = callback if callback else (lambda s,i: None)

    # Consume the initial iteration without updating the map for reporting.
    iterations = iter(regime)
    iteration = next(iterations)
    callback(som, iteration)

    # Process remaining iterations until the regime is completed. Literature
    # on the topic varies but consensus appears to be that the full training
    # set is processed per iteration of training.
    for iteration in iterations:
        callback(som, iteration)
        for vector in input_vectors:
            som.update(iteration, vector)

    return som


class Animator:
    '''
    Convenience class for generating matplotlib animations of self-organising
    maps. Supports multiple visualisations and chart titling.

    This class should be avoided in performance-sensitive activities.

    Author's note: This functionality is for generating animations in the
    accompanying notebook but also assisted debugging and exploring the
    training process.
    '''

    VISUALISATIONS = set([
        'weights',
        'best_units',
    ])

    def __init__(
        self,
        iteration_interval=1,
        visualisation=None,
        input_data=None
    ):
        '''
        Configures the animation. Note that if the iteration interval is
        greater than the number of iterations in the training process
        calling the animate method may result in unexpected behaviour.

        Args:
            iteration_interval: interval of iterations at which to generate
                animation frames
            visualisation: the visualisation to generate; one of:
                weights: the weights of the self-organising map
                best_units: activations of the best matching units
                    (also requires the inputs argument)
                all_units: like best_units but representing node
                    activation across the complete map
            input_data: inputs with which to visualise node activations

        Raises:
            ValueError: if the specified visualisation is not recognised
                or all required data for the visualisation is not provided
        '''
        if visualisation is None:
            self._visualisation = 'weights'
        else:
            if visualisation not in self.VISUALISATIONS:
                raise ValueError('Unrecogined visualisation')

            if visualisation != 'weights' and input_data is None:
                raise ValueError('\'input_data\' must not be none')

            self._visualisation = visualisation

        if iteration_interval < 0:
            raise ValueError('\'iteration_interval\' must be greater than zero')

        self._iteration_interval = iteration_interval

        self._input_data = input_data

        # Create the figure that will contain the animation.
        self._fig, self._ax = plt.subplots()

        # Collection of elements to track for animation.
        self._animate = []

        self._animation_opts = {
            'interval': 33,
            'blit': False,
            'repeat': False
        }

        self._title_opts = {
            'ha': 'center',
            'va': 'bottom',
            'color': 'k',
            'fontsize': 'large'
        }

    def frame(self, som, iteration=None):
        '''
        Visualises the self organising map.

        Args:
            som: the self-organising map to visualise
            iteration: the iteration represented in the frame

        Returns:
            List of matplotlib elements making up the visualisation, unless
            the iteration does not meet the interval iteration criterion in
            which case None is returned.
        '''
        if iteration and iteration.number % self._iteration_interval != 0:
            return None

        if self._visualisation == 'weights':
            image_data = som.nodes
        elif self._visualisation == 'best_units':
            # Track best matching units for each input data.
            bmus = {som.best_matching_unit(i): i for i in self._input_data}

            image_data = np.zeros(som.nodes.shape)
            for x in range(som.nodes.shape[0]):
                for y in range(som.nodes.shape[1]):
                    if (x,y) in bmus:
                        image_data[(x,y)] = bmus.get((x,y), 0)
        else:
            raise ValueError('Visualisation type escaped validation')

        image = self._ax.imshow(image_data, interpolation=None)

        # Iterations are necessary for frame titles; return early if the
        # title cannot be generated.
        if not iteration:
            return [image]

        # Using text for frame titles as matplotlib's titles do not seem to
        # work as expected with matplotlib animation.
        title_text = f'Self-organising map (iteration {iteration.number})'
        title = self._ax.text(0.5, 1.01, title_text,
                              transform=self._ax.transAxes,
                              **self._title_opts)
        return [image, title]

    def __call__(self, som, iteration):
        '''
        Generates a frame of animation for the given self-organising map.
        Intended to be a callback for create_som.

        Args:
            som: the self-organising map to visualise
            iteration: the iteration represented in the frame
        '''
        frame_elements = self.frame(som, iteration)
        if frame_elements:
            self._animate.append(frame_elements)

    def animate(self):
        '''
        Returns the animation as an HTML video. This method should be called
        once the process to visualise has completed.

        Returns:
            ArtistAnimation of generated frames.
        '''
        return animation.ArtistAnimation(self._fig, self._animate,
                                         **self._animation_opts)
