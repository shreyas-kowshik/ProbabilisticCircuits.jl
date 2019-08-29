[![Build Status](https://travis-ci.com/UCLA-StarAI/Juice.jl.svg?token=WqP1S31vh9msACoVUepf&branch=master)](https://travis-ci.com/UCLA-StarAI/Juice.jl)

# Juice.jl
Julia Circuit Empanada


# Requirements

Assuming you have Julia 1.1 installed, run the following to install the dependencies.

 ``julia install.jl``


# Documentation

To build the documentation locally, run the following to build the documentation, and then open `docs/build/index.html`.

    julia docs/make.jl




# Developement Tips

## Using Revise

`Revise.jl` allows you to modify code and use the changes without restarting Julia. For more information refer to [their github repository](https://github.com/timholy/Revise.jl).
 To use the Revise functionality while importing `Juice`, run the following command at the beginning of your REPL session. You may need to add additional files to track in `Debug.jl`.

    using Revise; include("test/Debug.jl");


## Sandbox

`Sandbox.jl` will contain some (old) examples on how to use the library.

## Testing

We will be adding automated tests under `/test` folder. To run all the test cases, run:

    julia test/runtests.jl

You can also run all the tests for a single module, for example:

    julia test/Juice/IO/runtests.jl

Or even any individual test, for example:

    julia test/Juice/IO/CircuitParserTest.jl

Make sure to run the tests before commiting new code.
