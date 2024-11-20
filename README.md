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

# Dataset otáčení:
Základní GAN o 4 vrstvách:

Testy:
Diskriminátor dosahuje po trenovaní zhruba 1000 epoch hodnot 0,a generátor hodnot 100. pokusím se trénovat diskriminátor s labelem místo 0,1 trochu více jemněji.

1) první možnost, co mě napadla je přidat gausovský šum na real_voxels, fake_voxels v diskriminátoru
přidání noisy_fake_voxels, noisy_real_voxels

po 600 epochách dostávám stejnou chybu - Nepomohlo

2) ZKusil jsem upravit learning late pro diskriminator, aby se více posnažil při trénování a stíhal s generátor.
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR * 0.5)

To trochu pomohlo, nicméně stále u 1500 epochy dostávám stejný problém - Nepomohlo.

3) Zkusil jsem dát dropout layery do diskriminátoru, které by mohli zabránit overfittingu - což mi pomohlo při bakalářce.
nn.Dropout(0.3)
Bohužel po 800 epochách vzniká stejná chyba - Nepomohlo

4) Dále jsem se pokusil volat generátor dvakrát při každém jednom volání diskriminátoru:
Bohužel se mi tohle nepodařilo implementovat, něco dělám zřejmě špatně.

5) Pak jsem ještě vyzkoušel snížit batch size na 16, zda to neovlivní trénovaní, nicmene to se přeučilo daleko dříve a to do 200 epoch, zkusil jsem tedy zvýšit. Zvýšením batch size na 128 jsem docílil, ze diskriminátor došel k nule později. až při 2500 epoše při hodnotě batch size 258. Avšak je zde stále ten samý problém. Je možné pak zkusit otestovat na ještě vyšších batch size, ale 258 mi přišlo už hodně samo o sobě.

6) Myslím si, že problém muže být způsobeným datasetem, který obsahuje jen točení kolem osy.

Zvýšil jsem dataset na 1000 a některé objekty opakuji, zároveň nebudu posouvat jen o jeden stupeň, ale o více. To se osvědčilo jako pomalý způsob jak trenink tak načítání + diskriminátor selhal po 100 epochách.

7) Zkusil jsem  zvýšit voxel resulution z 32x32x32 na 64x64x64, pro zachycení více detailu a snížil opět počet v datasetu na 100
input v diskriminatoru-> "nn.Linear(input_dim, 4096)," obdobně jsem udělal úpravy pro generátor a další části kodu.
Výsledkem po vizualizaci je, že bjekt je více přesnější, problémy s okny, které jsem měl zde vymizely.

Bohužel mam nedostatek paměti  Cuda jde out of memory - Zkusil jsem změnšit batch size na 8 místo 32 a nechat pouze 50 inputu z datasetu. - problém je pořad stejný -> musím vrátit zpátky na 32x32x32 :(.

Závěr:
Pořád si myslím, že problém je v datasetu a to, že samotné otáčení nestačí. 
Možné návrhy: posouvat offsetově po X ose, nevím zda tím otáčením nemátnu malinko tu sít -> třeba by stačilo pohyb okna a offset po ose x. 

Případně přidat další objekty do kolekce. 

Taktéž je možné, že tato základní neuronová sít to prostě nezvlada a chce to udelat jinou. Přesto bych očekával alespoň nějaký naptrný výsledek, nežli směs náhodně rozprostřených bločků.

Bylo by dobré se zároveň zaměřit na načítání souborů z obj do voxelu. Trvání této procedury je někdy i několik minut. Koukal jsme na možnosti a asi nejlepší by bylo, když bych vytvořil vždy cache soubor např. typu .npy, který mohu už načítat a nebudu se ztrácet čas při každém načítání-> tedy načítání obj by proběhlo jen jednou.


# Dataset - Pohyb okna:

Základní GAN se 4 vrstvami:
Hlavním problémem je velmi malá mřížka pro okna, což znamená, že okna nejsou dostatečně reprezentována – fakticky jde pouze o dva voxely, které tvoří celé okno. Přesto jsem při trénování dosáhl výrazně lepších výsledků ve srovnání s pouhým otáčením objektu. Zdá se, že síť se lépe učí i bez jakýchkoli úprav.

Během trénování, přibližně na 5000 epochách, však proces havaroval. Diskriminátor dosáhl hodnoty 100, zatímco generátor spadl na 0. Abych sledoval vývoj trénování, ukládal jsem váhy modelu každých 1000 epoch a použil je pro generování (vizualizaci) výsledků. Bohužel výsledný objekt je spíše chaotická změť náhodných voxelů, což se během dalších epoch nijak nezlepšovalo.

(Viz obrázek č. 1)

# Zjištění problému:
Výstupem každého voxelu by mělo být buď 0 (není), nebo 1 (je). Nicméně moje síť nepracuje s celými čísly, takže výsledkem je pravděpodobnost, že voxel existuje, což může zahrnovat i záporné hodnoty, i když velmi blízko nule.

Abych to vyřešil, nastavil jsem všechny hodnoty menší než 0,5 na 0, a ty ostatní ponechal. Možná by dávalo smysl přepočítat tyto hodnoty na 1, ale na výsledné chování by to nemělo mít zásadní vliv.

Dříve jsem zobrazoval všechny hodnoty větší než nula, což vedlo k tomu, že výstup tvořila směs náhodných voxelů. Když jsem tento výstup importoval do Blenderu, vygenerovala se plná krychle – všechny voxely byly nastaveny na hodnotu 1.

Přikládám obrázek č. 2 s aktuálním výstupem v Blenderu.

Plán na příští týden:
- posouvat budouvu  po x ose a různě s ní hýbat.
- Udelat  deformaci budovy, nebo přidat šum,případně přidat další budovy do datasetu, která budou obdobný
- Zkusit se podívat, zda neni nejaka voxelizace, co by mi vracela floaty.
