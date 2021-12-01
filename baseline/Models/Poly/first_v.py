import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FirstVertex(nn.Module):
    def __init__(self, opts, feats_dim, feats_channels):
        super(FirstVertex, self).__init__()
        self.grid_size = feats_dim
        self.opts = opts

        self.edge_conv = nn.Conv2d(
            in_channels = feats_channels,
            out_channels = 16,
            kernel_size = 3,
            padding = 1
        )

        self.edge_fc = nn.Linear(
            in_features = feats_dim**2 * 16,
            out_features = feats_dim**2
        )

        self.vertex_conv = nn.Conv2d(
            in_channels = feats_channels,
            out_channels = 16,
            kernel_size = 3,
            padding = 1
        )

        self.vertex_fc = nn.Linear(
            in_features = feats_dim**2 * 16,
            out_features = feats_dim**2
        )

    def forward(self, feats, temperature=0.0, beam_size=1):
        """
        if temperature < 0.01, use greedy
        else, use temperature
        """
        batch_size = feats.size(0)
        conv_edge = self.edge_conv(feats)
        conv_edge = F.relu(conv_edge, inplace=True)
        edge_logits = self.edge_fc(conv_edge.view(batch_size, -1))
        
        # Different from before, this used to take conv_edge as input before
        conv_vertex = self.vertex_conv(feats)
        conv_vertex = F.relu(conv_vertex)
        vertex_logits = self.vertex_fc(conv_vertex.view(batch_size, -1))
        logprobs = F.log_softmax(vertex_logits, -1)

        # Sample a first vertex
        if temperature < 0.01:
            logprob, pred_first = torch.topk(logprobs, beam_size, dim=-1)

        else:
            probs = torch.exp(logprobs/temperature)
            pred_first = torch.multinomial(probs, beam_size)

            # Get logprob of the sampled vertex
            logprob = logprobs.gather(1, pred_first)

        # Remove the last dimension if it is 1
        pred_first = torch.squeeze(pred_first, dim=-1)
        logprob = torch.squeeze(logprob, dim=-1)

        return edge_logits, vertex_logits, logprob, pred_first

    def first_point_correction(self, first_v, poly_class, thresh):
        first_v = first_v.unsqueeze(-1)
        poly_class = poly_class.to(torch.int64)

        x = first_v % self.grid_size
        y = first_v / self.grid_size
        # Each of shape batch_size, 1

        x_gt = poly_class % self.grid_size
        y_gt = poly_class / self.grid_size
        invalid = torch.eq(poly_class, self.grid_size**2)
        valid = torch.ne(poly_class, self.grid_size**2)
        # Each of shape batch_size, max_time_steps

        lengths = torch.sum(valid, dim=-1)

        dist = torch.abs(x_gt - x) + torch.abs(y_gt - y)
        # Manhattan Distance

        dist.masked_fill_(invalid, 10*self.grid_size)
        val, idx = torch.min(dist, dim=-1)

        new_poly_class = torch.zeros_like(poly_class, device=device)
        new_poly_class += self.grid_size**2
        # Fill with EOS
        
        for b in range(first_v.size(0)):
            # Across batches
            pos = idx[b]
            last_idx = lengths[b]
            if last_idx != self.opts['max_poly_len']:
                last_idx = last_idx - 1
                # Because we synthetically add the first
                # point at the end unless it's of max length
            
            new_poly_class[b, :last_idx] =\
            torch.cat((poly_class[b, pos:last_idx], poly_class[b, :pos]))

            if val[b] <= thresh:
                # No correction for first_v
                new_poly_class[b, 0] = first_v[b]

            if last_idx != self.opts['max_poly_len']:
                # Add first point to the end
                new_poly_class[b, last_idx] = new_poly_class[b, 0]

        return new_poly_class