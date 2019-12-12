u=-2:0.1:2;
y=[];
t=1;
for i=1:length(u)
    if abs(u(i))<t
        y=[y,0];
    else
        y=[y,abs(u(i))-t];
    end
end
plot(u,y,'k','linewidth',2)
hold on
plot(-t,0,'ok','markerfacecolor','w','linewidth',2)
plot(t,0,'ok','markerfacecolor','w','linewidth',2)
hold off
xticks([0])
yticks([0])
text(-t,0-.1,'-t')
text(t,0-.1,'t')
title('f(u)')
set(gcf,'color','w')