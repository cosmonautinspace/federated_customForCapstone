# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Libraries for the Shakespeare dataset for federated learning simulation."""

import collections
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import client_data
from tensorflow_federated.python.simulation.datasets import download
from tensorflow_federated.python.simulation.datasets import from_tensor_slices_client_data
from tensorflow_federated.python.simulation.datasets import sql_client_data


def _add_parsing(dataset: tf.data.Dataset) -> tf.data.Dataset:
  def _parse_example_bytes(serialized_proto_tensor):
    field_dict = {'snippets': tf.io.FixedLenFeature(shape=(), dtype=tf.string)}
    parsed_fields = tf.io.parse_example(serialized_proto_tensor, field_dict)
    return collections.OrderedDict(snippets=parsed_fields['snippets'])

  return dataset.map(_parse_example_bytes, num_parallel_calls=tf.data.AUTOTUNE)


def load_data(
    cache_dir: Optional[str] = None,
) -> tuple[client_data.ClientData, client_data.ClientData]:
  """Loads the federated Shakespeare dataset.

  Downloads and caches the dataset locally. If previously downloaded, tries to
  load the dataset from cache.

  This dataset is derived from the Leaf repository
  (https://github.com/TalwalkarLab/leaf) pre-processing on the works of
  Shakespeare, which is published in "LEAF: A Benchmark for Federated Settings"
  https://arxiv.org/abs/1812.01097.

  The data set consists of 715 users (characters of Shakespeare plays), where
  each
  example corresponds to a contiguous set of lines spoken by the character in a
  given play.

  Data set sizes:

  -   train: 16,068 examples
  -   test: 2,356 examples

  Rather than holding out specific users, each user's examples are split across
  _train_ and _test_ so that all users have at least one example in _train_ and
  one example in _test_. Characters that had less than 2 examples are excluded
  from the data set.

  The `tf.data.Datasets` returned by
  `tff.simulation.datasets.ClientData.create_tf_dataset_for_client` will yield
  `collections.OrderedDict` objects at each iteration, with the following keys
  and values:

    -   `'snippets'`: a `tf.Tensor` with `dtype=tf.string`, the snippet of
      contiguous text.

  Args:
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.

  Returns:
    Tuple of (train, test) where the tuple elements are
    `tff.simulation.datasets.ClientData` objects.
  """
  database_path = download.get_compressed_file(
      origin='https://storage.googleapis.com/tff-datasets-public/shakespeare.sqlite.lzma',
      cache_dir=cache_dir,
  )
  train_client_data = sql_client_data.SqlClientData(
      database_path, split_name='train'
  ).preprocess(_add_parsing)
  test_client_data = sql_client_data.SqlClientData(
      database_path, split_name='test'
  ).preprocess(_add_parsing)
  return train_client_data, test_client_data


def get_synthetic() -> client_data.ClientData:
  """Creates `tff.simulation.datasets.ClientData` for a synthetic in-memory example of Shakespeare.

  The returned `tff.simulation.datasets.ClientData` will have the same data
  schema as `load_data()`, but uses a very small set of client data loaded
  in-memory.

  This synthetic data is useful for validation in small tests.

  Returns:
    A `tff.simulation.datasets.ClientData` of synthentic Shakespeare text.
  """
  return from_tensor_slices_client_data.TestClientData(
      _SYNTHETIC_SHAKESPEARE_DATA
  )


# A small sub-sample of snippets from the Shakespeare dataset.
_SYNTHETIC_SHAKESPEARE_DATA = {
    'NLP_Notebook': collections.OrderedDict(
        snippets=[
            b'Natural Language Processing (NLP) stands at the crossroads of computer science, linguistics, and artificial intelligence, bridging the communication gap between humans and machines', b' It equips computers with the ability to grasp the intricacies and nuances of human language, from deciphering grammatical structures and semantic relationships to navigating the complexities of ambiguity, sarcasm, and cultural references', b' This empowers a vast array of applications that are revolutionizing how we interact with technology and extract meaning from the ever-growing deluge of textual and spoken data', b' At its core, NLP tackles the challenge of enabling computers to understand and interpret human language in all its richness', b' Rather than relying solely on literal interpretations, NLP techniques employ a multi-pronged approach', b' Rule-based systems provide a foundation by encoding grammatical structures and relationships within a language', b' However, their effectiveness is limited by the inherent fluidity and dynamism of natural language', b' This is where statistical and machine learning techniques come in, leveraging the power of AI', b' By training NLP systems on massive datasets of text and speech, they can learn the statistical patterns and relationships that underpin language', b' Machine learning, deep learning, and neural networks are particularly adept at handling these complexities', b' These capabilities fuel a range of core NLP functionalities', b' Natural Language Understanding (NLU) allows machines to decipher the intent and meaning behind language', b' This encompasses tasks like sentiment analysis, which pinpoints positive, negative, or neutral opinions; topic modeling, which uncovers underlying themes in text; and named entity recognition, which extracts specific entities like people, places, and organizations', b' By unlocking the meaning behind language, NLU empowers machines to make informed decisions and tailor responses accordingly', b' Natural Language Generation (NLG) flips the script, enabling machines to produce human-like text', b' This opens doors for applications like chatbots that can engage in conversations, machine translation that bridges language barriers, and content creation for marketing or summarization purposes', b' NLG algorithms strive to generate text that not only adheres to grammatical rules but also adapts its style to the specific context', b' The impact of NLP is rippling across a diverse range of industries and domains', b' Machine translation, constantly learning and improving through NLP algorithms, fosters seamless communication across languages, breaking down barriers and promoting global collaboration', b" Search engines leverage NLP to understand user queries with greater depth, returning results that align precisely with the user's intent", b' In the realm of customer service, NLP-powered chatbots can handle routine inquiries, efficiently resolve issues, and even personalize interactions, leading to enhanced customer satisfaction', b' Social media analysis tools powered by NLP extract valuable insights from conversations, helping businesses understand customer sentiment, track brand mentions, and identify emerging trends', b' NLP can even be harnessed for content creation, automating the generation of marketing materials, product descriptions, or even creative writing tasks, freeing up human resources for more strategic endeavors', b' As NLP continues to evolve, we can expect even more transformative applications to emerge on the horizon', b' From intelligent virtual assistants capable of nuanced conversations to automated document processing and analysis, NLP holds the potential to streamline workflows, augment human capabilities, and unlock entirely new avenues for human-computer interaction',
        ]
    ),
    'SMA_Notebook': collections.OrderedDict(
        snippets=[
            b'Social media analytics, the art of gathering and interpreting data from social media platforms, has become an essential tool for understanding online conversations and measuring the success of social media strategies', b' By delving into this rich pool of information, businesses and organizations can glean valuable insights about their audience, brand perception, and industry trends', b' Metrics like likes, comments, shares, and retweets provide a starting point, but social media analytics goes beyond vanity metrics', b' It digs deeper, using sentiment analysis to gauge audience feelings towards a brand or product, and topic modeling to uncover the underlying themes and conversations surrounding them', b' This allows for a nuanced understanding of what resonates with the audience and what areas need improvement', b' Social media analytics also plays a crucial role in influencer marketing, helping identify key opinion leaders and assess their impact on brand awareness and audience engagement', b' Competitive analysis is another area where social media analytics shines, enabling businesses to benchmark their performance against competitors and identify areas for differentiation', b' In essence, social media analytics empowers data-driven decision-making, allowing businesses to tailor their social media strategies for maximum impact and cultivate stronger relationships with their online communities',
        ]
    ),
    'ADS_Notebook': collections.OrderedDict(
        snippets=[
            b'Applied data science acts as the bridge between the theoretical underpinnings of data science and the practical world', b" It's not simply about understanding complex algorithms or statistical models; it's about wielding those tools to solve real-world problems across diverse domains", b' This field focuses on the execution of data-driven solutions, transforming raw data into actionable insights that can inform decision-making and drive positive outcomes', b' Central to applied data science is the data lifecycle', b' This encompasses the entire journey of data, from its initial collection and cleaning to its analysis, visualization, and ultimately, its deployment in a solution', b' Applied data scientists possess a strong understanding of each stage, ensuring data quality, accuracy, and relevance for the problem at hand', b' The toolbox of applied data science is vast, drawing upon a multitude of techniques and technologies', b' Statistical analysis remains a cornerstone, enabling the extraction of meaningful patterns and relationships from data', b' Machine learning algorithms, particularly those adept at supervised and unsupervised learning, play a central role in building predictive models and uncovering hidden patterns', b' Data visualization is crucial for translating complex data insights into clear, compelling narratives that resonate with stakeholders', b' Programming languages like Python and R are the workhorses, allowing data scientists to manipulate, analyze, and model data effectively', b' The applications of applied data science are truly boundless', b' In finance, it empowers risk assessment, fraud detection, and personalized investment recommendations', b' Healthcare leverages applied data science for disease prediction, drug discovery, and personalized medicine', b' Businesses utilize it for customer segmentation, targeted marketing campaigns, and market research', b' Scientific research benefits from applied data science in areas like climate modeling, materials science, and drug development', b' Ultimately, the success of applied data science hinges on collaboration', b' Data scientists work closely with domain experts, business analysts, and stakeholders throughout the process', b' This ensures that the solutions developed are not only technically sound but also meet the specific needs and goals of the domain', b' As the volume and complexity of data continue to surge, applied data science will undoubtedly play an increasingly critical role in driving innovation and progress across various sectors',
        ]
    ),
}
