siRNA_Names=C3_(:,1);
AssociatedGeneStrings=C3_(:,2);
No_of_AssociatedGenes=zeros(4000,1);
for i=1:4000
    No_of_AssociatedGenes(i)=count(AssociatedGeneStrings(i),';');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

AssociatedGenes = cell(4000, 1);
for i=1:4000
  if No_of_AssociatedGenes>0
      AssociatedGenes{i}=strsplit(AssociatedGeneStrings(i),';');
  else
      AssociatedGenes{i}=AssociatedGeneStrings(i);
  end
end