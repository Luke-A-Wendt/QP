u=-2:0.1:2;
y=[];
t=1;
for i=1:length(u)
    if abs(u(i))<t
        y=[y,u(i)^2];
    else
        y=[y,t*(2*abs(u(i))-t)];
    end
end
plot(u,y,'k','linewidth',2)
hold on
plot([-t,t],[t^2,t^2],'--k')
plot([t,t],[0,t^2],'--k')
plot([-t,-t],[0,t^2],'--k')
plot(t,t^2,'ok','markerfacecolor','w','linewidth',2)
plot(-t,t^2,'ok','markerfacecolor','w','linewidth',2)
hold off
xticks([0])
yticks([0])
text(-t,0-.2,'-t')
text(t,0-.2,'t')
text(-2-.3,t^2,'t^2')
title('f(u)')
set(gcf,'color','w')