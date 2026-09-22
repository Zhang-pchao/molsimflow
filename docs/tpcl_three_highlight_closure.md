# TPCL three-highlight closure

`molsimflow.postprocess.tpcl_three_highlight_closure` combines the accepted
Stage C review, prior three-highlight synthesis, Stage A observable
qualification, and Stage A2 robustness audit into a final existing-trajectory
decision package.

The allowed closure labels are:

- `CLOSED_PUBLICATION_GRADE`
- `CLOSED_WITH_LIMITS`
- `MECHANISM_NOT_ESTABLISHED`
- `REQUIRES_NEW_MD`

The synthesis does not submit MD. Its causal rule is strict: parameter
sensitivity and non-overlapping blocks from one parent history are not
independent histories.
