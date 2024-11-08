# Metody generování 3D prostředí založené na strojovém učení
## 21.10-27.10
Nenašel jsem dataset s 3D daty pro trénování mé sítě.
1) Vzít poly-low objekt z netu a rozšířit si ho.
2) Vzít objekty např. z minecraftu a vyexportovat je jako .obj
Po počátečním průzkumu datasetů jsem se rozhodl vytvořit vlastní sadu dat tím, že budu generovat variace z jednoho objektu z netu pomocí augmentace.

Snaha o generaci objektu skrze čistý Python - bez úspěchu.

## 28.10-3.11
Nainstaloval jsem si Blender, kde jsem se seznámil se základními funkcemi a vytvořil skript, který mi umožňuje otáčet objekty kolem Y osy a exportovat je jako .obj.

Objekt sice otáčím, ale myslím si, že to nebude dostatečný, tedy jsem vyříznul okno v budově objektu, udělat ho jako samostatný objekt a následně udělat skript pro náhodné umistování po ploše budovy. 

## 4.10-10.11
Vyřešil jsem některé problémy s oknem
  - objekt window překryl objektem building - boolean aplikace
  - vytvářím kopie objektů, které po skončení cyklu odstraňuju
  - spojil jsem ukládání u rotace s pohybem okna, tedy nyní ukládám variace dle potřeby.
  - udělal jsme nějakou základní gan sít, která pracuje s 3d objekty, přestože zatím pořad modely převádím zle
    
- [x] - Počátčení trénování sítě? => Problém s přenosem obj do voxel podoby.
- [x] -  spojení obou skriptů dohromady?
- [ ] -  vyřešení umistování objektu po všech stranách? => zatím není nutné, nejprve se budu soustředit na sít, inputy nejake mam. 
- [ ] - odstranění nějakým způsobem okno, které bylo původně na budově? => stejná odpověd, jako s umístěním po všech stranach budovy. Tohle rozšíření mužu udělat, až začnu dobře trénovat.

