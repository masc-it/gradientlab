# Context

You are an expert computer vision research scientist.

The research topic is: text compression in form of images and neural decoder.

A dataset of images is created by drawing text into a 128x128px image, with black background and white utf glyphs.

An estimate says that a given 128x128px image can contain at most L=1024 chars, but of course this can vary according to the font used and canvas size.

The vocabulary is a padded UTF vocab with size V of 256 UTF glyphs + "<|pad|>" + "<|im_end|>", for a total of 512 tokens. It's basically an extended byte level tokenizer. which is already available at `gradientlab.tokenizers.byte_tokenizer` and uses HF interfaces.


# The idea

The idea I want to prototype is to have a CNN encoder network that encodes an image and then treats the output as a fixed sequence (of length L) of glyph slots. Each slot will try to classify the input feature in one of the vocab entries (V=512).

I'd like to use cross attention between the CNN feature map output and the output of size LxV. So the network can build L semantic queries, before producing the final glyph.

Moreover, a "sequence counting" regressor head is trained in parallel and can help when the user want to perform actual generations in inference.

The design idea is that the encoder has to learn good features, it has to do most of the work, then cross attention creates a contextual brigde to feed the final slots classifier.

# Task

Let's develop "ImageTextSlot" a CNN network that takes in a grayscale image of NxN pixels (128 by default) and predicts for each of the N slots, the associated utf glyph code.

The model is made of a state-of-art CNN encoder (from scratch), cross attentions and a fused linear that predicts the slots and a linear for the character slots counting.

At the end, produce a "HOWTO.md" file with instructions of how to instantiate the model, run a forward showing how the input tensors have to be setup, loss functions, how to run predictions and how to prepare the dataset.

# Instructions and code structure

- Your experiment working directory is `exp20251230_imagetextzip`, all the other experiment folders are irrelevant.
- You can already find a trainer.py you can adapt freely
- Model code has to be all written under modeling/ folder
- modeling/factory.py has to be used to create model, tokenizer and cfg with parameters setup.
- main.py is the entrypoint
- exp_config.py contains the experiment hyperparams
- the project is configured with uv, so use uv to run eventual python commands
- you can find nn building blocks under "neuralblocks" module
- other utils are under "training_utils", "tokenizers", "logging_utils".

## Network and trainer

- Provide a proper weight init function for the model
- use pixel_values as the image input tensor and labels as the actual labels tensor
- the dataset used during training will return a dict with the inputs the network needs, so just a basic collate_fn will be needed.
