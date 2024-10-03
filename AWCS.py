class AWCS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  padding=1, group=16):
        super().__init__()
        
        self.softmax = nn.Softmax(dim=-1)
        
        
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        

        self.ds_conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size, stride=2, padding=padding)
        
   
        self.us_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1)

    def forward(self, x):
       
      
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
      
        att = self.softmax(att)
      
     
        x = self.ds_conv(x)
      
  
        x = rearrange(x,'bs (s ch) h w -> bs ch h w s',s=4)
        x = torch.sum(x * att, dim=-1)
        
  
        x = self.us_conv(x)
        
        return x
