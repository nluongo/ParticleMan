To implement:
- Add event-level features (data/MC, physics process, etc)
- Add truth/reco flag for individual particles
- Per-feature masking: Allow masking individual features (pT, eta, phi) independently
  rather than entire particles. Concatenate a binary mask vector [m_pT, m_eta, m_phi] to the embedding
  input so the MLP can learn to ignore masked features while using unmasked ones as context.
- Proper phi handling, replace with sin(phi), cos(phi)
- Implement attention bias for delta\_phi, delta\_eta, delta\_R, remove phi from standard embedding

