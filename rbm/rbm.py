#
# class BaseRBM(object):
#     def __init__(self):
#         pass
#
#     def _propup(self, v):
#         pass
#
#     def _propdown(self, h):
#         pass
#
#     def _means_h_given_v(self, h_means):
#         pass
#
#     def _sample_h_given_v(self, v):
#         pass
#
#     def _means_v_given_h(self, h):
#         pass
#
#     def _sample_v_given_h(self, v_means):
#         pass
#
#     def _gibbs_step(self, h):
#         pass
#
#     def _gibb_chain(self, h):
#         pass
#
#     def _train_step(self,batch_x):
#         h0_means = self._means_h_given_v(batch_x)
#         h0_samples = self._sample_h_given_v(h0_means)
#         h0_states = h0_samples
#
#         v1_states, v1_means, h1_states, h1_means = self._gibb_chain(h0_states)
#
#         dW_positive = torch.mul(batch_x.t(),h0_means)
#         dW_negative = torch.mul(v1_states.t(),h1_means)
#         dW =