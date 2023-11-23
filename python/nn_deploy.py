# copied from cmsml package

import os
import tensorflow as tf
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.eager.def_function import Function
from tensorflow.python.eager.function import ConcreteFunction
from tensorflow.python.framework import convert_to_constants

def save_frozen_graph(path, obj, variables_to_constants=False, *args, **kwargs):
  """
  Extracts a TensorFlow graph from an object *obj* and saves it at *path*. The graph is optionally
  transformed into a simpler representation with all its variables converted to constants when
  *variables_to_constants* is *True*. The saved file contains the graph as a protobuf. The
  accepted types of *obj* greatly depend on the available API versions.

  *obj* can be a compiled keras model, or either a polymorphic or
  concrete function as returned by ``tf.function``. Polymorphic functions either must have a
  defined input signature (``tf.function(input_signature=(...,))``) or they must accept no
  arguments in the first place. See the TensorFlow documentation on `concrete functions
  <https://www.tensorflow.org/guide/concrete_function>`__ for more info.

  *args* and *kwargs* are forwarded to ``tf.io.write_graph``.
  """
  path = os.path.expandvars(os.path.expanduser(str(path)))
  graph_dir, graph_name = os.path.split(path)

  # default as_text value
  kwargs.setdefault("as_text", path.endswith((".pbtxt", ".pb.txt")))

  # convert keras models and polymorphic functions to concrete functions, v2 only
  if isinstance(obj, tf.keras.Model):
    model_func = saving_utils.trace_model_call(obj)
    if model_func.function_spec.arg_names and not model_func.input_signature:
      raise ValueError("when obj is a keras model callable accepting arguments, its "
                       "input signature must be frozen by building the model")
    obj = model_func.get_concrete_function()
  elif isinstance(obj, Function):
    if obj.function_spec.arg_names and not obj.input_signature:
      raise ValueError("when obj is a polymorphic function accepting arguments, its "
                       "input signature must be frozen")
    obj = obj.get_concrete_function()

  # convert variables to constants
  if variables_to_constants:
    if not isinstance(obj, ConcreteFunction):
      raise TypeError("when variables_to_constants is true, obj must be a concrete "
                      f"or polymorphic function, got '{obj}' instead")
    obj = convert_to_constants.convert_variables_to_constants_v2(obj)

  # extract the graph
  graph = obj.graph if isinstance(obj, ConcreteFunction) else obj

  # write it
  tf.io.write_graph(graph, graph_dir, graph_name, *args, **kwargs)

def load_frozen_graph(path, as_text=None):
  """
  Reads a saved TensorFlow graph from *path* and returns it. When *as_text* is either *True* or *None*,
  and the file extension is ``".pbtxt"`` or ``".pb.txt"``, the content of the file at *path* is
  expected to be a human-readable text file. Otherwise, it is read as a binary protobuf file.
  Example:

  .. code-block:: python
      graph = load_frozen_graph("path/to/model.pb")
  """
  path = os.path.expandvars(os.path.expanduser(str(path)))

  # default as_text value
  if as_text is None:
    as_text = path.endswith((".pbtxt", ".pb.txt"))

  graph = tf.Graph()
  with graph.as_default():
    graph_def = graph.as_graph_def()
    if as_text:
      # use a simple pb reader to load the file into graph_def
      from google.protobuf import text_format
      with open(path, "rb") as f:
          text_format.Merge(f.read(), graph_def)
    else:
      # use the gfile api depending on the TF version
      with tf.io.gfile.GFile(path, "rb") as f:
          graph_def.ParseFromString(f.read())

    # import the graph_def (pb object) into the actual graph
    tf.import_graph_def(graph_def, name="")

  return graph
