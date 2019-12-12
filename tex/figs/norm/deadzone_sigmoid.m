t=2;
u=-10*t:0.1:10*t;
y=[];
for i=1:length(u)
    if abs(u(i))<t
        y=[y,0];
    else
        y=[y,sign(u(i))-t/u(i)];
    end
end
plot(u,y,'k','linewidth',2)
hold on
% plot([-t,t],[t^2,t^2],'--k')
% plot([t,t],[0,t^2],'--k')
% plot([-t,-t],[0,t^2],'--k')
plot(t,0,'ok','markerfacecolor','w','linewidth',2)
plot(-t,0,'ok','markerfacecolor','w','linewidth',2)
hold off
xticks([0])
yticks([-1,0,1])
text(-t-.5,-1-.075,'-t')
text(t,-1-.075,'t')
% text(-2-.3,t^2,'t^2')
xlabel('u')
ylabel('y')
title('f(u)/u')
set(gcf,'color','w')