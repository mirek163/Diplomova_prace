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

## 11.11.-14.11
Pořád řeším převod objektu na správné rozměry. Převádím objekt z formátu .obj do voxel formátu. Vybrán voxel dle rešerše místo cloud z mé výzkumné práce. Objekt se správně překládá do Voxel podoby, sít nicméně vyžaduje rozměry 32x32x32. Využíval jsem funkci zoom, jelikož muj objekt nemá stejné velikosti x,y,z a tímto bych měl být schopen udělat lineární interpolaci. Bohužel zde narážím na problém, že objekt je například na jedné stráně oříznut. Pokusil jsem se udělat tyto úpravy v blenderu. Rozměry 32x32x32 jsem sice nastavil, ale potýkal jsem se s tím, že síť velmi dlouho trvala a nakonec jsem to musel ukončit. pokusil jsem se převést objekt na různé variance a to vydělením 32, nastavení rozměrů 0,32x0,32x0,32 atp. bohužel bez úspěchu. Zkusím se ještě podívat více na funkci zoom, dle mého by to mělo dělat přesně co potřebuji.

Pro účely testování jsem udělal funkci na jednoduché zobrazování objketu skrze matplotlib.

## 18.11-24.11
Ï. Implementoval jsem ukládání vah jak pro generátor tak diskriminátor. Možné pokračování po předložení těch vah. V rámci tohohle jsem implementoval, pokud selže program nebo ho ukončím "stopem", dostanu vahy (weight_latest) pro případ, že by se to rozhodlo selhat ke konci trénování. Funkce pro generování .obj souboru na základě vah z generátoru a v poslední řadě úpravu načítání obj souborů a jejich převod do voxel podoby - vizualizace pomocí baru a určení kolik zbývá.


